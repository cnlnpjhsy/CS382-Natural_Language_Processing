#coding=utf8
import os, json
PAD = '<pad>'
UNK = '<unk>'
BOS = '<s>'
EOS = '</s>'


class Vocab():
    '创建编码词表，将字转换为嵌入或者将嵌入转换为字。'
    def __init__(self, padding=False, unk=False, min_freq=1, filepath=None):
        super(Vocab, self).__init__()
        # 创建双向的转换字典
        self.word2id = dict()
        self.id2word = dict()
        # 有padding和unk时，[0]表示PAD，[1]表示UNK
        if padding:
            idx = len(self.word2id)
            self.word2id[PAD], self.id2word[idx] = idx, PAD
        if unk:
            idx = len(self.word2id)
            self.word2id[UNK], self.id2word[idx] = idx, UNK

        if filepath is not None:
            self.from_train(filepath, min_freq=min_freq)

    def from_train(self, filepath, min_freq=1):
        with open(filepath, 'r', encoding='utf-8') as f:
            trains = json.load(f)
        # 创建字频字典：{字: 出现次数}，最小次数为1
        word_freq = {}
        for data in trains:     # trains有多个data
            for utt in data:    # 每个data有多个utt
                text = utt['asr_1best']     # 从噪声文本获取字
                for char in text:   # 将字的出现频率存入字频字典
                    word_freq[char] = word_freq.get(char, 0) + 1
        for word in word_freq:
            if word_freq[word] >= min_freq:
                # 对词频字典内的字，依次添加至转换字典中
                idx = len(self.word2id)
                self.word2id[word], self.id2word[idx] = idx, word

    def __len__(self):
        return len(self.word2id)

    @property
    def vocab_size(self):
        return len(self.word2id)

    def __getitem__(self, key):
        return self.word2id.get(key, self.word2id[UNK])


class LabelVocab():
    '创建标注表，将标注转换为嵌入表示形式。'
    def __init__(self, root):
        self.tag2idx, self.idx2tag = {}, {}
        # 设置标注的PAD->0，'O'->1。'O'表示不属于任何类型
        self.tag2idx[PAD] = 0
        self.idx2tag[0] = PAD
        self.tag2idx['O'] = 1
        self.idx2tag[1] = 'O'
        self.from_filepath(root)

    def from_filepath(self, root):
        ontology = json.load(open(os.path.join(root, 'ontology.json'), 'r', encoding='utf-8'))
        acts = ontology['acts']     # 两种动作
        slots = ontology['slots']   # 多种语义槽

        # 组合每一种动作、语义槽与BIO标注，构成标注表
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
