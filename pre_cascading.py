import json


def load_data(file):
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    records = []
    for line in lines:
        record = json.loads(line)
        records.append(record)
    return records


def write(data, fn):
    with open(fn, 'w', encoding='utf-8') as f:
        for line in data:
            line = json.dumps(line, ensure_ascii=False)
            f.write(line + '\n')


TYPES = ['起诉', '投资', '减持', '股份股权转让', '质押', '收购', '判决', '签署合同', '担保', '中标']


def process(records):
    data_len = len(records)
    data_new = []
    for i in range(data_len):
        record = records[i]
        data_id = record['id']
        events = record['events']
        content = record['content']

        # label all occurrence types
        type_occur = []
        for TYPE in TYPES:
            for event in events:
                event_type = event['type']
                if event_type == TYPE:
                    type_occur.append(TYPE)
        type_occur = list(set(type_occur))

        # label triggers and arguments
        for TYPE in TYPES:
            events_typed = []
            for event in events:
                event_type = event['type']
                if event_type == TYPE:
                    events_typed.append(event)
            # label triggers
            if len(events_typed) != 0:
                triggers = []
                trigger_args = {}
                for event in events_typed:
                    trigger = event['trigger']['span']
                    if trigger not in triggers:
                        triggers.append(trigger)
                    trigger_args[str(trigger)] = trigger_args.get(str(trigger), {})
                    for arg_role in event['args']:
                        trigger_args[str(trigger)][arg_role] = trigger_args[str(trigger)].get(arg_role, [])
                        args_roled_spans = [item['span'] for item in event['args'][arg_role]]
                        for args_roled_span in args_roled_spans:
                            if args_roled_span not in trigger_args[str(trigger)][arg_role]:
                                trigger_args[str(trigger)][arg_role].append(args_roled_span)
                # according to trigger order, write json record
                triggers_str = [str(trigger) for trigger in triggers]  # with order
                for trigger_str in trigger_args:
                    index = triggers_str.index(trigger_str)
                    data_dict = {}
                    data_dict['id'] = data_id
                    data_dict['content'] = content
                    data_dict['occur'] = type_occur
                    data_dict['type'] = TYPE
                    data_dict['triggers'] = triggers
                    data_dict['index'] = index
                    data_dict['args'] = trigger_args[trigger_str]
                    data_new.append(data_dict)
    return data_new


# {"id": "9e573f697633ad200c9aa86d70bc2103", "content": "此外,隆鑫通用的公告还透露,除了中信证券,隆鑫控股与四川信托有限公司、中国国际金融有限公司、国民信托有限公司的股票质押合同均已到期,但“经与质权人友好协商,目前暂不会做违约处置”。", "events": [
# {"type": "质押", "trigger": {"span": [57, 59], "word": "质押"}, "args": {"sub-org": [{"span": [3, 7], "word": "隆鑫通用"}], "obj-org": [{"span": [16, 20], "word": "中信证券"}], "collateral": [{"span": [55, 57], "word": "股票"}]}},
# {"type": "质押", "trigger": {"span": [57, 59], "word": "质押"}, "args": {"sub-org": [{"span": [3, 7], "word": "隆鑫通用"}], "obj-org": [{"span": [26, 34], "word": "四川信托有限公司"}], "collateral": [{"span": [55, 57], "word": "股票"}]}},
# {"type": "质押", "trigger": {"span": [57, 59], "word": "质押"}, "args": {"sub-org": [{"span": [3, 7], "word": "隆鑫通用"}], "obj-org": [{"span": [35, 45], "word": "中国国际金融有限公司"}], "collateral": [{"span": [55, 57], "word": "股票"}]}},
# {"type": "质押", "trigger": {"span": [57, 59], "word": "质押"}, "args": {"sub-org": [{"span": [3, 7], "word": "隆鑫通用"}], "obj-org": [{"span": [46, 54], "word": "国民信托有限公司"}], "collateral": [{"span": [55, 57], "word": "股票"}]}}]}


def main():
    train = load_data('./datasets/FewFC/data/train.json')
    train = process(train)
    write(train, './datasets/FewFC/cascading_sampled/train.json')

    dev = load_data('./datasets/FewFC/data/dev.json')
    dev = process(dev)
    write(dev, './datasets/FewFC/cascading_sampled/dev.json')

    test = load_data('./datasets/FewFC/data/test.json')
    test = process(test)
    write(test, './datasets/FewFC/cascading_sampled/test.json')


if __name__ == '__main__':
    main()
