import json
import os
from pypinyin import pinyin, lazy_pinyin, Style

DATAROOT = './data'
input = os.path.join(DATAROOT, 'lexicon/poi_name.txt')
output = os.path.join(DATAROOT, 'poi_name_ngram.json')

with open(input, 'r', encoding='utf-8') as f:
    poi_list = f.readlines()

poi_2gram, poi_3gram, poi_4gram = {}, {}, {}
for poi in poi_list:
    poi = poi.strip()
    if len(poi) >= 2:
        poi_2 = poi[:2]
        index = ' '.join(lazy_pinyin(poi_2, style=Style.TONE3))
        if index not in poi_2gram:
            poi_2gram[index] = []
        poi_2gram[index].append(poi)
    if len(poi) >= 3:
        poi_3 = poi[:3]
        index = ' '.join(lazy_pinyin(poi_3, style=Style.TONE3))
        if index not in poi_3gram:
            poi_3gram[index] = []
        poi_3gram[index].append(poi)
    if len(poi) >= 4:
        poi_4 = poi[:4]
        index = ' '.join(lazy_pinyin(poi_4, style=Style.TONE3))
        if index not in poi_4gram:
            poi_4gram[index] = []
        poi_4gram[index].append(poi)

file = {
    '2-grams': poi_2gram,
    # '2-grams': {
    #     'shang4 hai3': ['上海交通大学', '上海站'],
    #     'bei3 jing1': ['北京', '北京市'],
    #     ...
    # },
    '3-grams': poi_3gram,
    '4-grams': poi_4gram
}
json.dump(file, open(output, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
print('2-grams: %d, 3-grams: %d, 4-grams: %d' % (len(poi_2gram), len(poi_3gram), len(poi_4gram)))
