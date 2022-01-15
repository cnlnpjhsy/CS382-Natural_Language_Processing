#-*- coding:utf-8 -*-
import torch

from utils.example import Example
from utils.vocab import VocabTokenizer


def from_example_list(args, ex_list, device='cpu', train=True):
    '获取Example的列表。每一个utt（有噪声文本）都是一个Example。bert不再需要输入按文本长度降序排列。'
    batch = Batch(ex_list, device)
    tag_pad_idx = args.tag_pad_idx

    batch.utt = [ex.utt for ex in ex_list]
    batch.utt_split = [ex.utt_split for ex in ex_list]
    # 将utt文本成批送入Tokenizer，可以自动填充PAD。转换为张量，作为模型的输入
    input_idx, input_type_idx, input_attn_mask = batch.tokenizer(batch.utt_split)
    batch.input_idx = input_idx.to(device)
    batch.input_type_idx = input_type_idx.to(device)
    batch.input_attn_mask = input_attn_mask.to(device)

    if train:   # 训练时，为batch添加正确的标签
        # 获取batch里Example的'动作-语义槽-槽值'标注
        batch.labels = [ex.slotvalue for ex in ex_list]
        # 获取batch里Example的意图标注
        intents = [ex.intent for ex in ex_list]
        # 获取batch里各个标注的'B/I-动作-语义槽'嵌入表示列表。以最长的那个文本为最大长度，不足用PAD补齐
        tag_lens = [len(ex.tag_id) for ex in ex_list]
        max_tag_lens = max(tag_lens)
        tag_ids = [ex.tag_id + [tag_pad_idx] * (max_tag_lens - len(ex.tag_id)) for ex in ex_list]
        # 将PAD用掩码掩盖掉
        tag_mask = [[1] * len(ex.tag_id) + [0] * (max_tag_lens - len(ex.tag_id)) for ex in ex_list]
        # 把batch里统一长度的Example词标注列表转换为张量，用于训练
        batch.intents = torch.tensor(intents, dtype=torch.long, device=device)
        batch.tag_ids = torch.tensor(tag_ids, dtype=torch.long, device=device)
        batch.tag_mask = torch.tensor(tag_mask, dtype=torch.float, device=device)
    else:
        batch.labels = None
        batch.intents = None
        batch.tag_ids = None
        batch.tag_mask = None

    return batch


class Batch():

    def __init__(self, examples, device):
        super(Batch, self).__init__()

        self.examples = examples
        self.tokenizer = VocabTokenizer()
        self.device = device

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]