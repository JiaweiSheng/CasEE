import json


def score(preds_tuple, golds_tuple):
    '''
    Modified from https://github.com/xinyadu/eeqa
    '''
    gold_mention_n, pred_mention_n, true_positive_n = 0, 0, 0
    for sentence_id in golds_tuple:
        gold_sentence_mentions = golds_tuple[sentence_id]
        pred_sentence_mentions = preds_tuple[sentence_id]
        gold_sentence_mentions = set(gold_sentence_mentions)
        pred_sentence_mentions = set(pred_sentence_mentions)
        for mention in pred_sentence_mentions:
            pred_mention_n += 1
        for mention in gold_sentence_mentions:
            gold_mention_n += 1
        for mention in pred_sentence_mentions:
            if mention in gold_sentence_mentions:
                true_positive_n += 1
    prec_c, recall_c, f1_c = 0, 0, 0
    if pred_mention_n != 0:
        prec_c = true_positive_n / pred_mention_n
    else:
        prec_c = 0
    if gold_mention_n != 0:
        recall_c = true_positive_n / gold_mention_n
    else:
        recall_c = 0
    if prec_c or recall_c:
        f1_c = 2 * prec_c * recall_c / (prec_c + recall_c)
    else:
        f1_c = 0
    return prec_c, recall_c, f1_c


def gen_tuples(record):
    if record:
        ti, tc, ai, ac = [], [], [], []
        for event in record:
            typ, trigger_span = event['type'], event['trigger']['span']
            ti_one = (trigger_span[0], trigger_span[1])
            tc_one = (typ, trigger_span[0], trigger_span[1])
            ti.append(ti_one)
            tc.append(tc_one)
            for arg_role in event['args']:
                for arg_role_one in event['args'][arg_role]:
                    ai_one = (typ, arg_role_one['span'][0], arg_role_one['span'][1])
                    ac_one = (typ, arg_role_one['span'][0], arg_role_one['span'][1], arg_role)

                    ai.append(ai_one)
                    ac.append(ac_one)
        return ti, tc, ai, ac
    else:
        return [], [], [], []


def cal_scores_ti_tc_ai_ac(preds, golds):
    '''
    :param preds: {id: [{type:'', 'trigger':{'span':[], 'word':[]}, args:[role1:[], role2:[], ...}, ...]}
    :param golds:
    :return:
    '''
    # assert len(preds) == len(golds)
    tuples_pred = [{}, {}, {}, {}]  # ti, tc, ai, ac
    tuples_gold = [{}, {}, {}, {}]  # ti, tc, ai, ac

    for idx in golds:
        if idx not in preds:
            pred = None
        else:
            pred = preds[idx]
        gold = golds[idx]

        ti, tc, ai, ac = gen_tuples(pred)
        tuples_pred[0][idx] = ti
        tuples_pred[1][idx] = tc
        tuples_pred[2][idx] = ai
        tuples_pred[3][idx] = ac

        ti, tc, ai, ac = gen_tuples(gold)
        tuples_gold[0][idx] = ti
        tuples_gold[1][idx] = tc
        tuples_gold[2][idx] = ai
        tuples_gold[3][idx] = ac

    prf_s = []
    for i in range(4):
        prf = score(tuples_pred[i], tuples_gold[i])
        prf_s.append(prf)
    return prf_s


def cal_scores(pred_dict, gold_dict, print_tab=False):
    prf_s = cal_scores_ti_tc_ai_ac(pred_dict, gold_dict)
    metric_names = ['TI', 'TC', 'AI', 'AC']
    for i, prf in enumerate(prf_s):
        if not print_tab:
            print('{}: P:{:.1f}, R:{:.1f}, F:{:.1f}'.format(metric_names[i], prf[0] * 100, prf[1] * 100, prf[2] * 100))
        else:
            print('{}:\tP:\t{:.1f}\tR:\t{:.1f}\tF:\t{:.1f}'.format(metric_names[i], prf[0] * 100, prf[1] * 100, prf[2] * 100))
    return prf_s


def gen_idx_event_dict(records):
    data_dict = {}
    for line in records:
        idx = line['id']
        events = line['events']
        data_dict[idx] = events
    return data_dict
