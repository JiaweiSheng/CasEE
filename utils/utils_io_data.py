import json


def read_json(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def read_jsonl(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(json.loads(line))
    return data


def write_json(data, fn):
    with open(fn, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


def write_jsonl(data, fn):
    with open(fn, 'w', encoding='utf-8') as f:
        for line in data:
            line = json.dumps(line, ensure_ascii=False)
            f.write(line + '\n')