import json
import os
import re
from pypinyin import lazy_pinyin, Style
from dimsim import get_distance as pinyin_distance
from Levenshtein import distance as edit_distance

PINYIN_RECALL_DISTANCE = 15.0
EDIT_RECALL_DISTANCE = 5.0
ACCEPT_DISTANCE = 15.0

class Correction():
    def __init__(self, root):
        self.poi_database = []
        self.grams2 = {}
        self.grams3 = {}
        self.grams4 = {}
        self.from_filepath(root)

        self.history = []

    def from_filepath(self, root):
        poi_database = open(os.path.join(root, 'lexicon/poi_name.txt'), 'r', encoding='utf-8').readlines()
        self.poi_database = [poi.strip() for poi in poi_database]

        ngrams = json.load(open(os.path.join(root, 'poi_name_ngram.json'), 'r', encoding='utf-8'))
        self.grams2 = ngrams['2-grams']
        self.grams3 = ngrams['3-grams']
        self.grams4 = ngrams['4-grams']

    def tag_is_poi(self, str):
        return re.match(r'.*-(poi|起点|终点|途经点).*-.*', str) is not None

    def recall_poi(self, pinyin_dict, pinyin, candidate={}):
        poi_candidate = candidate   # {poi: 距离}，例：{'上海': 0.13, ...}
        for key in pinyin_dict:
            dist = pinyin_distance(pinyin, key.split())
            if dist <= PINYIN_RECALL_DISTANCE:
                for poi in pinyin_dict[key]:
                    if poi not in poi_candidate:
                        poi_candidate[poi] = dist
                    else:
                        poi_candidate[poi] = min(dist, poi_candidate[poi])
        return poi_candidate

    def save_correction_history(self, output_path):
        json.dump(self.history, open(output_path, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)

    def __call__(self, utt, pred):
        for i, tag in enumerate(pred):
            if self.tag_is_poi(tag):
                tag_tuple = tag.split('-')
                value = tag_tuple[2]
                if value in self.poi_database:
                    continue
                # 识别出来的poi槽值不在数据库中，尝试进行修正
                if len(value) >= 5:
                    # 槽值长度>=5时，截取前4个字的拼音，去4-grams寻找
                    index = lazy_pinyin(value[:4], style=Style.TONE3)
                    candidate = self.recall_poi(self.grams4, index)
                elif len(value) == 4:
                    # 槽值长度为4，分别在3-grams和4-grams寻找
                    index = lazy_pinyin(value[:4], style=Style.TONE3)
                    candidate = self.recall_poi(self.grams4, index)
                    index = lazy_pinyin(value[:3], style=Style.TONE3)
                    candidate = self.recall_poi(self.grams3, index, candidate)
                elif len(value) == 3:
                    # 槽值长度为3，分别在2-grams和3-grams寻找
                    index = lazy_pinyin(value[:3], style=Style.TONE3)
                    candidate = self.recall_poi(self.grams3, index)
                    index = lazy_pinyin(value[:2], style=Style.TONE3)
                    candidate = self.recall_poi(self.grams2, index, candidate)
                elif len(value) == 2:
                    # 槽值长度为2，在2-grams寻找
                    index = lazy_pinyin(value[:2], style=Style.TONE3)
                    candidate = self.recall_poi(self.grams2, index)
                else:
                    # 槽值长度不足2，认为是特殊情况或者标注错误，不作处理
                    continue
                
                for poi in candidate:
                    dist = edit_distance(value, poi)
                    if dist > EDIT_RECALL_DISTANCE:
                        dist = ACCEPT_DISTANCE
                    candidate[poi] += dist

                best_poi = min(candidate.keys(), key=(lambda poi: candidate[poi]))
                if candidate[best_poi] <= ACCEPT_DISTANCE:
                    tag_tuple[2] = best_poi
                    pred[i] = '-'.join(tag_tuple)
                    self.history.append({
                        'asr_text': utt, 
                        'original value': value, 
                        'correction value': best_poi, 
                        'distance': candidate[best_poi]
                        })
        return pred
