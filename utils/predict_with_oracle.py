import torch
import numpy as np


def extract_specific_item_with_oracle(model, d_t, token, seg, mask, rp, tm, args_num, threshold_0, threshold_1, threshold_2, threshold_3, threshold_4, ty_args_id):
    assert token.size(0) == 1
    data_type = d_t.item()
    text_emb = model.plm(token, seg, mask)

    # predict event type
    p_type, type_emb = model.predict_type(text_emb, mask)
    type_pred = np.array(p_type > threshold_0, dtype=int)
    type_rep = type_emb[d_t, :]

    # predict event trigger
    p_s, p_e, text_rep_type = model.predict_trigger(type_rep, text_emb, mask)
    trigger_s = np.where(p_s > threshold_1)[0]
    trigger_e = np.where(p_e > threshold_2)[0]
    trigger_spans = []
    for i in trigger_s:
        es = trigger_e[trigger_e >= i]
        if len(es) > 0:
            e = es[0]
            trigger_spans.append((i, e))

    # predict event argument
    p_s, p_e, type_soft_constrain = model.predict_args(text_rep_type, rp, tm, mask, type_rep)
    p_s = np.transpose(p_s)
    p_e = np.transpose(p_e)
    args_spans = {i: [] for i in range(args_num)}
    for i in ty_args_id[data_type]:
        args_s = np.where(p_s[i] > threshold_3)[0]
        args_e = np.where(p_e[i] > threshold_4)[0]
        for j in args_s:
            es = args_e[args_e >= j]
            if len(es) > 0:
                e = es[0]
                args_spans[i].append((j, e))
    return type_pred, trigger_spans, args_spans


def predict_one(model, args, typ_truth, token, seg, mask, r_p, t_m, tri_truth, args_truth, ty_args_id, typ_oracle, tri_oracle):
    type_pred, trigger_pred, args_pred = extract_specific_item_with_oracle(model, typ_oracle, token, seg, mask, r_p, t_m, args.args_num, args.threshold_0, args.threshold_1, args.threshold_2, args.threshold_3, args.threshold_4, ty_args_id)
    type_oracle = typ_oracle.item()
    type_truth = typ_truth.view(args.type_num).cpu().numpy().astype(int)
    trigger_truth, args_truth = tri_truth[0], args_truth[0]

    # used to save tuples, which is like:
    trigger_pred_tuples = []  # (type, tri_sta, tri_end), 3-tuple
    trigger_truth_tuples = []
    args_pred_tuples = []  # (type, tri_sta, tri_end, arg_sta, arg_end, arg_role), 6-tuple
    args_truth_tuples = []

    for trigger_pred_one in trigger_pred:
        typ = type_oracle
        sta = trigger_pred_one[0]
        end = trigger_pred_one[1]
        trigger_pred_tuples.append((typ, sta, end))

    for trigger_truth_one in trigger_truth:
        typ = type_oracle
        sta = trigger_truth_one[0]
        end = trigger_truth_one[1]
        trigger_truth_tuples.append((typ, sta, end))

    args_candidates = ty_args_id[type_oracle]  # type constrain
    for i in args_candidates:
        typ = type_oracle
        tri_sta = tri_oracle[0]
        tri_end = tri_oracle[1]
        arg_role = i
        for args_pred_one in args_pred[i]:
            arg_sta = args_pred_one[0]
            arg_end = args_pred_one[1]
            args_pred_tuples.append((typ, arg_sta, arg_end, arg_role))

        for args_truth_one in args_truth[i]:
            arg_sta = args_truth_one[0]
            arg_end = args_truth_one[1]
            args_truth_tuples.append((typ, arg_sta, arg_end, arg_role))

    return type_pred, type_truth, trigger_pred_tuples, trigger_truth_tuples, args_pred_tuples, args_truth_tuples
