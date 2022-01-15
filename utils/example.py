import json

from utils.vocab import CLS, SEP, VocabTokenizer, LabelVocab
from utils.word2vec import Word2vecUtils
from utils.evaluator import Evaluator

class Example():

    @classmethod
    def configuration(cls, root):
        cls.evaluator = Evaluator()
        # cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path)
        # cls.word2vec = Word2vecUtils(word2vec_path)
        cls.vocab_tokenizer = VocabTokenizer()
        cls.label_vocab = LabelVocab(root)

    @classmethod
    def load_dataset(cls, data_path):
        '将数据集中的每个utt都以Example类的形式保存下来，并返回它们的列表。'
        datas = json.load(open(data_path, 'r', encoding='utf-8'))
        examples = []
        for data in datas:
            for utt in data:
                ex = cls(utt, nolabel=False)
                examples.append(ex)
        return examples

    @classmethod
    def load_testset(cls, data_path):
        datas = json.load(open(data_path, 'r', encoding='utf-8'))
        examples = []
        for data in datas:
            for utt in data:
                ex = cls(utt, nolabel=True)
                examples.append(ex)
        return examples


    def __init__(self, ex: dict, nolabel=False):
        '''
        构造函数，每一个utt都是一个Example。
        这个构造函数可以对有噪声文本进行BIO标注，并得到该文本的嵌入表示与文本标注的嵌入表示。
        例如：帮(O)我(O)导(B-inform-操作)航(I-inform-操作)
        标注的方法是使用无噪声的槽值对有噪声文本进行匹配。有匹配才有标注。
        '''
        super(Example, self).__init__()
        self.ex = ex

        self.utt = ex['asr_1best'].lower().replace(' ', '')     # bert不识别空格；标签的字母为小写
        self.utt_split = list(self.utt)
        if not nolabel:
        # 这个utt的槽字典，{'动作-语义槽': 槽值}
            self.slot = {}
            self.intent = [0, 0]
            self.intent2idx = {
                'inform': 0,
                'deny': 1
                }
            for label in ex['semantic']:    # 获取语义标注
                # 将动作inform, deny视作句子意图
                self.intent[self.intent2idx[label[0]]] = 1
                # act_slot = '动作-语义槽'，以标注作为槽字典的索引
                act_slot = f'{label[0]}-{label[1]}'
                if len(label) == 3:
                    self.slot[act_slot] = label[2]  # 添加槽值
            self.tags = ['O'] * len(self.utt)   # 标注初始化：噪声文本的每个字都标注为'O'
            for slot in self.slot:      # 对槽字典内的每个标注：
                value = self.slot[slot]
                bidx = self.utt.find(value)     # 在噪声文本中寻找无噪声槽值的匹配
                if bidx != -1:      # 若找到，对噪声文本中的槽值进行BI标注
                    # 对每个字的标注形式为：O 或者 B-动作-语义槽 或者 I-动作-语义槽
                    self.tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)
                    self.tags[bidx] = f'B-{slot}'
            # 为语义标注增加CLS和SEP
            self.tags = [CLS] + self.tags + [SEP]
            # 这个utt的槽列表，['动作-语义槽-槽值']。例如：'inform-操作-导航'
            self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]
            # 将噪声文本的标注转换为标注嵌入形式的列表
            l = Example.label_vocab
            self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]
