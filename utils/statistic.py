import json
import os

DATAROOT = './data'
input = os.path.join(DATAROOT, 'test.json')
output = os.path.join(DATAROOT, 'test_statistic.json')

correct_count, error_count = 0, 0
correct_utt, error_utt = [], []

datas = json.load(open(input, 'r', encoding='utf-8'))
for data in datas:
    for utt in data:

        if sorted(utt['semantic']) == sorted(utt['pred']):
            correct_count += 1
            correct_utt.append(utt)
        else:
            error_count += 1
            error_utt.append(utt)

file = [
    {
        'correct_count': correct_count, 
        'utt': correct_utt
    },
    {
        'error_count': error_count,
        'utt': error_utt
    }
]
json.dump(file, open(output, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
