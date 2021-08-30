import os
import time
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from utils.utils_io_model import load_model, save_model
import torch
import numpy as np
from sklearn.metrics import *
from utils.predict_without_oracle import extract_all_items_without_oracle
from utils.predict_with_oracle import predict_one
from tqdm import tqdm
from utils.metric import score, gen_idx_event_dict, cal_scores, cal_scores_ti_tc_ai_ac
from utils.utils_io_data import read_jsonl, write_jsonl


class Framework(object):
    def __init__(self, config, model):
        self.config = config
        self.model = model.to(config.device)

    def load_model(self, model_path):
        self.model = load_model(self.model, model_path)

    def set_learning_setting(self, config, train_loader, dev_loader, model):
        instances_num = len(train_loader.dataset)
        train_steps = int(instances_num * config.epochs_num / config.batch_size) + 1

        print("Batch size: ", config.batch_size)
        print("The number of training instances:", instances_num)
        print("The number of evaluating instances:", len(dev_loader.dataset))

        bert_params = list(map(id, model.bert.parameters()))

        other_params = filter(lambda p: id(p) not in bert_params, model.parameters())
        optimizer_grouped_parameters = [{'params': model.bert.parameters()}, {'params': other_params, 'lr': config.lr_task}]

        optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr_bert, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=train_steps * config.warmup, num_training_steps=train_steps)

        if config.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=config.fp16_opt_level)

        if torch.cuda.device_count() > 1:
            print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(model)

        return scheduler, optimizer

    def train(self, train_loader, dev_loader):
        scheduler, optimizer = self.set_learning_setting(self.config, train_loader, dev_loader, self.model)
        # going to train
        total_loss = 0.0
        ed_loss = 0.0
        te_loss = 0.0
        ae_loss = 0.0
        best_f1 = 0.0
        best_epoch = 0
        for epoch in range(1, self.config.epochs_num + 1):
            print('Training...')
            self.model.train()
            for i, (idx, d_t, t_v, token, seg, mask, t_index, r_pos, t_m, t_s, t_e, a_s, a_e, a_m) in enumerate(train_loader):
                self.model.zero_grad()
                d_t = torch.LongTensor(d_t).to(self.config.device)
                t_v = torch.FloatTensor(t_v).to(self.config.device)
                token = torch.LongTensor(token).to(self.config.device)
                seg = torch.LongTensor(seg).to(self.config.device)
                mask = torch.LongTensor(mask).to(self.config.device)
                r_pos = torch.LongTensor(r_pos).to(self.config.device)
                t_m = torch.LongTensor(t_m).to(self.config.device)
                t_s = torch.FloatTensor(t_s).to(self.config.device)
                t_e = torch.FloatTensor(t_e).to(self.config.device)
                a_s = torch.FloatTensor(a_s).to(self.config.device)
                a_e = torch.FloatTensor(a_e).to(self.config.device)
                a_m = torch.LongTensor(a_m).to(self.config.device)
                loss, type_loss, trigger_loss, args_loss = self.model(token, seg, mask, d_t, t_v, t_s, t_e, r_pos, t_m, a_s, a_e, a_m)
                if torch.cuda.device_count() > 1:
                    loss = torch.mean(loss)
                    type_loss = torch.mean(type_loss)
                    trigger_loss = torch.mean(trigger_loss)
                    args_loss = torch.mean(args_loss)

                total_loss += loss.item()
                ed_loss += type_loss.item()
                te_loss += trigger_loss.item()
                ae_loss += args_loss.item()

                if (i + 1) % self.config.report_steps == 0:
                    print("Epoch id: {}, Training steps: {}, ED loss:{:.6f},TE loss:{:.6f}, AE loss:{:.6f},  Avg loss: {:.6f}".format(epoch, i + 1, ed_loss / self.config.report_steps, te_loss / self.config.report_steps, ae_loss / self.config.report_steps,
                                                                                                                                      total_loss / self.config.report_steps))
                    total_loss = 0.0
                    ed_loss = 0.0
                    te_loss = 0.0
                    ae_loss = 0.0
                if self.config.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()
                scheduler.step()

            print('Evaluating...')
            c_ps, c_rs, c_fs, t_ps, t_rs, t_fs, a_ps, a_rs, a_fs = self.evaluate_with_oracle(self.config, self.model, dev_loader, self.config.device, self.config.ty_args_id, self.config.id_type)
            f1_mean_all = (c_fs + t_fs + a_fs) / 3
            print('Evaluate on all types:')
            print("Epoch id: {}, Type P: {:.3f}, Type R: {:.3f}, Type F: {:.3f}".format(epoch, c_ps, c_rs, c_fs))
            print("Epoch id: {}, Trigger P: {:.3f}, Trigger R: {:.3f}, Trigger F: {:.3f}".format(epoch, t_ps, t_rs, t_fs))
            print("Epoch id: {}, Args P: {:.3f}, Args R: {:.3f}, Args F: {:.3f}".format(epoch, a_ps, a_rs, a_fs))
            print("Epoch id: {}, F1 Mean All: {:.3f}".format(epoch, f1_mean_all))

            if f1_mean_all > best_f1:
                best_f1 = f1_mean_all
                best_epoch = epoch
                save_model(self.model, self.config.output_model_path)
            print("The Best F1 Is: {:.3f}, When Epoch Is: {}".format(best_f1, best_epoch))

    def evaluate_with_oracle(self, config, model, dev_data_loader, device, ty_args_id, id2type):
        if hasattr(model, "module"):
            model = model.module
        model.eval()
        # since there exists "an" idx with "several" records, we use dict to combine the results
        type_pred_dict = {}
        type_truth_dict = {}
        trigger_pred_tuples_dict = {}
        trigger_truth_tuples_dict = {}
        args_pred_tuples_dict = {}
        args_truth_tuples_dict = {}

        for i, (idx, typ_oracle, typ_truth, token, seg, mask, t_index, r_p, t_m, tri_truth, args_truth) in tqdm(enumerate(dev_data_loader)):
            typ_oracle = torch.LongTensor(typ_oracle).to(device)
            typ_truth = torch.FloatTensor(typ_truth).to(device)
            token = torch.LongTensor(token).to(device)
            seg = torch.LongTensor(seg).to(device)
            mask = torch.LongTensor(mask).to(device)
            r_p = torch.LongTensor(r_p).to(device)
            t_m = torch.LongTensor(t_m).to(device)

            tri_oracle = tri_truth[0][t_index[0]]
            type_pred, type_truth, trigger_pred_tuples, trigger_truth_tuples, args_pred_tuples, args_truth_tuples = predict_one(model, config, typ_truth, token, seg, mask, r_p, t_m, tri_truth, args_truth, ty_args_id, typ_oracle, tri_oracle)

            idx = idx[0]
            # collect type predictions
            if idx not in type_pred_dict:
                type_pred_dict[idx] = type_pred
            if idx not in type_truth_dict:
                type_truth_dict[idx] = type_truth

            # collect trigger predictions
            if idx not in trigger_pred_tuples_dict:
                trigger_pred_tuples_dict[idx] = []
            trigger_pred_tuples_dict[idx].extend(trigger_pred_tuples)
            if idx not in trigger_truth_tuples_dict:
                trigger_truth_tuples_dict[idx] = []
            trigger_truth_tuples_dict[idx].extend(trigger_truth_tuples)

            # collect argument predictions
            if idx not in args_pred_tuples_dict:
                args_pred_tuples_dict[idx] = []
            args_pred_tuples_dict[idx].extend(args_pred_tuples)
            if idx not in args_truth_tuples_dict:
                args_truth_tuples_dict[idx] = []
            args_truth_tuples_dict[idx].extend(args_truth_tuples)

        # Here we calculate event detection metric (macro).
        type_pred_s, type_truth_s = [], []
        for idx in type_truth_dict.keys():
            type_pred_s.append(type_pred_dict[idx])
            type_truth_s.append(type_truth_dict[idx])
        type_pred_s = np.array(type_pred_s)
        type_truth_s = np.array(type_truth_s)
        c_ps = precision_score(type_truth_s, type_pred_s, average='macro')
        c_rs = recall_score(type_truth_s, type_pred_s, average='macro')
        c_fs = f1_score(type_truth_s, type_pred_s, average='macro')

        # Here we calculate TC and AC metric with oracle inputs.
        t_p, t_r, t_f = score(trigger_pred_tuples_dict, trigger_truth_tuples_dict)
        a_p, a_r, a_f = score(args_pred_tuples_dict, args_truth_tuples_dict)
        return c_ps, c_rs, c_fs, t_p, t_r, t_f, a_p, a_r, a_f

    def evaluate_without_oracle(self, config, model, data_loader, device, seq_len, id_type, id_args, ty_args_id):
        if torch.cuda.device_count() > 1:
            model = model.module
        model.eval()
        results = []
        for i, (idx, content, token, seg, mask) in tqdm(enumerate(data_loader)):
            idx = idx[0]
            token = torch.LongTensor(token).to(device)
            seg = torch.LongTensor(seg).to(device)
            mask = torch.LongTensor(mask).to(device)
            result = extract_all_items_without_oracle(model, device, idx, content, token, seg, mask, seq_len, config.threshold_0, config.threshold_1, config.threshold_2, config.threshold_3, config.threshold_4, id_type, id_args, ty_args_id)
            results.append(result)
        pred_records = results
        pred_dict = gen_idx_event_dict(pred_records)
        gold_records = read_jsonl(self.config.test_path)
        gold_dict = gen_idx_event_dict(gold_records)
        prf_s = cal_scores_ti_tc_ai_ac(pred_dict, gold_dict)
        return prf_s, pred_records
