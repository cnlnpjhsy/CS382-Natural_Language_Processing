#coding=utf8
import os, json
from transformers import AutoTokenizer
PAD = '<pad>'
UNK = '<unk>'
CLS = '<s>'
SEP = '</s>'


class VocabTokenizer():
    '用于将输入句子转换为bert token形式。'
    def __init__(self, tokenizer_path):
        super(VocabTokenizer, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def __call__(self, utt):
        token = self.tokenizer(utt, padding='longest', return_tensors='pt', is_split_into_words=True)
        token_idx = token['input_ids']
        type_idx = token['token_type_ids']
        masks = token['attention_mask']
        return token_idx, type_idx, masks


class LabelVocab():
    '创建标注表，将标注转换为嵌入表示形式。'
    def __init__(self, root):
        self.tag2idx, self.idx2tag = {}, {}
        # 设置标注的PAD->0，'O'->1。'O'表示不属于任何类型
        self.tag2idx[PAD] = 0
        self.idx2tag[0] = PAD
        self.tag2idx['O'] = 1
        self.idx2tag[1] = 'O'
        # 对bert，增加CLS和SEP两种额外标注
        self.tag2idx[CLS] = 2
        self.idx2tag[2] = CLS
        self.tag2idx[SEP] = 3
        self.idx2tag[3] = SEP
        self.from_filepath(root)

    def from_filepath(self, root):
        ontology = json.load(open(os.path.join(root, 'ontology.json'), 'r', encoding='utf-8'))
        acts = ontology['acts']     # 2种动作
        slots = ontology['slots']   # 10种语义槽

        # 组合每一种动作、语义槽与BI标注，构成标注表
        for act in acts:
            for slot in slots:
                for bi in ['B', 'I']:
                    idx = len(self.tag2idx)
                    tag = f'{bi}-{act}-{slot}'
                    self.tag2idx[tag], self.idx2tag[idx] = idx, tag

    def convert_tag_to_idx(self, tag):
        return self.tag2idx[tag]

    def convert_idx_to_tag(self, idx):
        return self.idx2tag[idx]

    @property
    def num_tags(self):
        return len(self.tag2idx)
