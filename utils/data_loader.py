from torch.utils.data import Dataset
import json
import numpy as np
import os


def get_dict(fn):
    with open(fn + '/cascading_sampled/ty_args.json', 'r', encoding='utf-8') as f:
        ty_args = json.load(f)
    if not os.path.exists(fn + '/cascading_sampled/shared_args_list.json'):
        args_list = set()
        for ty in ty_args:
            for arg in ty_args[ty]:
                args_list.add(arg)
        args_list = list(args_list)
        with open(fn + '/cascading_sampled/shared_args_list.json', 'w', encoding='utf-8') as f:
            json.dump(args_list, f, ensure_ascii=False)
    else:
        with open(fn + '/cascading_sampled/shared_args_list.json', 'r', encoding='utf-8') as f:
            args_list = json.load(f)

    args_s_id = {}
    args_e_id = {}
    for i in range(len(args_list)):
        s = args_list[i] + '_s'
        args_s_id[s] = i
        e = args_list[i] + '_e'
        args_e_id[e] = i

    id_type = {i: item for i, item in enumerate(ty_args)}
    type_id = {item: i for i, item in enumerate(ty_args)}

    id_args = {i: item for i, item in enumerate(args_list)}
    args_id = {item: i for i, item in enumerate(args_list)}
    ty_args_id = {}
    for ty in ty_args:
        args = ty_args[ty]
        tmp = [args_id[a] for a in args]
        ty_args_id[type_id[ty]] = tmp
    return type_id, id_type, args_id, id_args, ty_args, ty_args_id, args_s_id, args_e_id


def read_labeled_data(fn):
    ''' Read Train Data / Dev Data '''
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data_ids = []
    data_content = []
    data_type = []
    data_occur = []
    data_triggers = []
    data_index = []
    data_args = []
    for line in lines:
        line_dict = json.loads(line.strip())
        data_ids.append(line_dict.get('id', 0))
        data_occur.append(line_dict['occur'])
        data_type.append(line_dict['type'])
        data_content.append(line_dict['content'])
        data_index.append(line_dict['index'])
        data_triggers.append(line_dict['triggers'])
        data_args.append(line_dict['args'])
    return data_ids, data_occur, data_type, data_content, data_triggers, data_index, data_args


def read_unlabeled_data(fn):
    ''' Read Test Data'''
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data_ids = []
    data_content = []
    for line in lines:
        line_dict = json.loads(line.strip())
        data_ids.append(line_dict['id'])
        data_content.append(line_dict['content'])
    return data_ids, data_content


def get_relative_pos(start_idx, end_idx, length):
    '''
    return relative position 
    [start_idx, end_idx]
    '''
    pos = list(range(-start_idx, 0)) + [0] * (end_idx - start_idx + 1) + list(range(1, length - end_idx))
    return pos


def get_trigger_mask(start_idx, end_idx, length):
    '''
    used to generate trigger mask, where the element of start/end postion is 1
    [000010100000]
    '''
    mask = [0] * length
    mask[start_idx] = 1
    mask[end_idx] = 1
    return mask


class Data(Dataset):
    def __init__(self, task, fn, tokenizer=None, seq_len=None, args_s_id=None, args_e_id=None, type_id=None):
        assert task in ['train', 'eval_with_oracle', 'eval_without_oracle']
        self.task = task
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.args_s_id = args_s_id
        self.args_e_id = args_e_id
        self.args_num = len(args_s_id.keys())
        self.type_id = type_id
        self.type_num = len(type_id.keys())

        if self.task == 'eval_without_oracle':
            data_ids, data_content = read_unlabeled_data(fn)
            self.data_ids = data_ids
            self.data_content = data_content
            tokens_ids, segs_ids, masks_ids = self.data_to_id(data_content)

            self.token = tokens_ids
            self.seg = segs_ids
            self.mask = masks_ids

        else:
            data_ids, data_occur, data_type, data_content, data_triggers, data_index, data_args = read_labeled_data(fn)
            self.data_ids = data_ids
            self.data_occur = data_occur
            self.data_triggers = data_triggers
            self.data_args = data_args

            self.data_content = data_content
            tokens_ids, segs_ids, masks_ids = self.data_to_id(data_content)
            self.token = tokens_ids
            self.seg = segs_ids
            self.mask = masks_ids

            data_type_id_s, type_vec_s = self.type_to_id(data_type, data_occur)
            self.data_type_id_s = data_type_id_s
            self.type_vec_s = type_vec_s

            self.r_pos, self.t_m = self.get_rp_tm(data_triggers, data_index)
            self.t_index = data_index

            if self.task == 'train':
                t_s, t_e = self.trigger_seq_id(data_triggers)
                self.t_s = t_s
                self.t_e = t_e
                a_s, a_e, a_m = self.args_seq_id(data_args)
                self.a_s = a_s
                self.a_e = a_e
                self.a_m = a_m

            if self.task == 'eval_with_oracle':
                self.data_content = data_content
                self.data_args = data_args
                self.data_triggers = data_triggers
                triggers_truth_s, args_truth_s = self.results_for_eval()
                self.triggers_truth = triggers_truth_s
                self.args_truth = args_truth_s

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, index):
        if self.task == 'train':
            return self.data_ids[index], \
                   self.data_type_id_s[index], \
                   self.type_vec_s[index], \
                   self.token[index], \
                   self.seg[index], \
                   self.mask[index], \
                   self.t_index[index], \
                   self.r_pos[index], \
                   self.t_m[index], \
                   self.t_s[index], \
                   self.t_e[index], \
                   self.a_s[index], \
                   self.a_e[index], \
                   self.a_m[index]
        elif self.task == 'eval_with_oracle':
            return self.data_ids[index], \
                   self.data_type_id_s[index], \
                   self.type_vec_s[index], \
                   self.token[index], \
                   self.seg[index], \
                   self.mask[index], \
                   self.t_index[index], \
                   self.r_pos[index], \
                   self.t_m[index], \
                   self.triggers_truth[index], \
                   self.args_truth[index]
        elif self.task == 'eval_without_oracle':
            return self.data_ids[index], \
                   self.data_content[index], \
                   self.token[index], \
                   self.seg[index], \
                   self.mask[index]
        else:
            raise Exception('task not define !')

    def data_to_id(self, data_contents):
        tokens_ids = []
        segs_ids = []
        masks_ids = []
        for i in range(len(self.data_ids)):
            data_content = data_contents[i]
            # default uncased
            data_content = [token.lower() for token in data_content]
            data_content = list(data_content)
            # Here we add <CLS> and <SEP> token for BERT input
            # transformers == 4.9.1
            inputs = self.tokenizer.encode_plus(data_content, add_special_tokens=True, max_length=self.seq_len, truncation=True, padding='max_length')
            tokens, segs, masks = inputs["input_ids"], inputs["token_type_ids"], inputs['attention_mask']
            tokens_ids.append(tokens)
            segs_ids.append(segs)
            masks_ids.append(masks)
        return tokens_ids, segs_ids, masks_ids

    def type_to_id(self, data_type, data_occur):
        data_type_id_s, type_vec_s = [], []
        for i in range(len(self.data_ids)):
            data_type_id = self.type_id[data_type[i]]
            type_vec = np.array([0] * self.type_num)
            for occ in data_occur[i]:
                idx = self.type_id[occ]
                type_vec[idx] = 1
            data_type_id_s.append(data_type_id)
            type_vec_s.append(type_vec)
        return data_type_id_s, type_vec_s

    def trigger_seq_id(self, data_triggers):
        '''
        given trigger span, return ground truth trigger matrix, for bce loss
        t_s: trigger start sequence, 1 for position 0
        t_e: trigger end sequence, 1 for position 0
        '''
        trigger_s = []
        trigger_e = []
        for i in range(len(self.data_ids)):
            data_trigger = data_triggers[i]
            t_s = [0] * self.seq_len
            t_e = [0] * self.seq_len

            for t in data_trigger:
                # plus 1 for additional <CLS> token
                t_s[t[0] + 1] = 1
                t_e[t[1] + 1 - 1] = 1

            trigger_s.append(t_s)
            trigger_e.append(t_e)
        return trigger_s, trigger_e

    def args_seq_id(self, data_args_list):
        '''
        given argument span, return ground truth argument matrix, for bce loss
        '''
        args_s_lines = []
        args_e_lines = []
        arg_masks = []
        for i in range(len(self.data_ids)):
            args_s = np.zeros(shape=[self.args_num, self.seq_len])
            args_e = np.zeros(shape=[self.args_num, self.seq_len])
            data_args_dict = data_args_list[i]
            arg_mask = [0] * self.args_num
            for args_name in data_args_dict:
                s_r_i = self.args_s_id[args_name + '_s']
                e_r_i = self.args_e_id[args_name + '_e']
                arg_mask[s_r_i] = 1
                for span in data_args_dict[args_name]:
                    # plus 1 for additional <CLS> token
                    args_s[s_r_i][span[0] + 1] = 1
                    args_e[e_r_i][span[1] + 1 - 1] = 1
            args_s_lines.append(args_s)
            args_e_lines.append(args_e)
            arg_masks.append(arg_mask)
        return args_s_lines, args_e_lines, arg_masks

    def results_for_eval(self):
        '''
        read structured ground truth, for evaluating model performance
        '''
        triggers_truth_s = []
        args_truth_s = []
        for i in range(len(self.data_ids)):
            triggers = self.data_triggers[i]
            args = self.data_args[i]
            # plus 1 for additional <CLS> token
            triggers_truth = [(span[0] + 1, span[1] + 1 - 1) for span in triggers]
            args_truth = {i: [] for i in range(self.args_num)}
            for args_name in args:
                s_r_i = self.args_s_id[args_name + '_s']
                for span in args[args_name]:
                    # plus 1 for additional <CLS> token
                    args_truth[s_r_i].append((span[0] + 1, span[1] + 1 - 1))
            triggers_truth_s.append(triggers_truth)
            args_truth_s.append(args_truth)
        return triggers_truth_s, args_truth_s

    def get_rp_tm(self, triggers, data_index):
        '''
        get relative position embedding and trigger mask, according to the trigger span 
        r_pos: relation position embedding
        t_m: trigger mask, used for mean pooling
        '''
        r_pos = []
        t_m = []
        for i in range(len(self.data_ids)):
            trigger = triggers[i]
            index = data_index[i]
            span = trigger[index]
            # plus 1 for additional <CLS> token
            pos = get_relative_pos(span[0] + 1, span[1] + 1 - 1, self.seq_len)
            pos = [p + self.seq_len for p in pos]
            # plus 1 for additional <CLS> token
            mask = get_trigger_mask(span[0] + 1, span[1] + 1 - 1, self.seq_len)
            r_pos.append(pos)
            t_m.append(mask)
        return r_pos, t_m


def collate_fn_train(data):
    '''
    :param data: [(x, y), (x, y), (), (), ()]
    :return:
    idx: the id of data record
    dt: the type of event (str)
    t_v: 
    token: token sequence
    seg: segment sequence
    mask: mask sequence
    t_index: unused; used to indicate the trigger number of argument   
    r_pos: relative position embedding
    t_m: trigger_mask，where 1 for start/end postion
    t_s, ground_truth of trigger start
    t_e, ground_truth of trigger end
    a_s, ground_truth of argument start
    a_e, ground_truth of argument end
    a_m, unused; used to indicate the correlation between argument role and event type.
    '''
    idx, dt, t_v, token, seg, mask, t_index, r_pos, t_m, t_s, t_e, a_s, a_e, a_m = zip(*data)
    return idx, dt, t_v, token, seg, mask, t_index, r_pos, t_m, t_s, t_e, a_s, a_e, a_m


def collate_fn_dev(data):
    '''
    :param data: [(x, y), (x, y), (), (), ()]
    :return:
    '''
    idx, dt, t_v, token, seg, mask, t_index, r_pos, t_m, t_t, a_t = zip(*data)
    return idx, dt, t_v, token, seg, mask, t_index, r_pos, t_m, t_t, a_t


def collate_fn_test(data):
    '''
    :param data: [(x, y), (x, y), (), (), ()]
    :return:
    '''
    idx, dc, token, seg, mask = zip(*data)
    return idx, dc, token, seg, mask
