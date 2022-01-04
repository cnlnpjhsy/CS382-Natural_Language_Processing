#-*- coding:utf-8 -*-
import torch

from utils.example import Example


def from_example_list(args, ex_list, device='cpu', train=True):
    '获取Example的列表。每一个utt（有噪声文本）都是一个Example。按照Example噪声文本长度的降序排列。'
    ex_list = sorted(ex_list, key=lambda x: len(x.input_idx), reverse=True)
    batch = Batch(ex_list, device)
    if not train:
        list_idx = []
        for ex in ex_list:
            list_idx.append(ex.original_idx)
        batch.get_original_idx(list_idx)
    pad_idx = args.pad_idx
    tag_pad_idx = args.tag_pad_idx

    batch.utt = [ex.utt for ex in ex_list]
    # 以batch里最长的那个文本为最大长度，将不足的用PAD补齐
    input_lens = [len(ex.input_idx) for ex in ex_list]
    max_len = max(input_lens)
    input_ids = [ex.input_idx + [pad_idx] * (max_len - len(ex.input_idx)) for ex in ex_list]
    # 最后将batch里统一长度的Example词idx列表转换为张量
    batch.input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    batch.lengths = input_lens

    if train:   # 训练时，为batch添加正确的标签
        # 获取batch里Example的'动作-语义槽-槽值'标注
        batch.labels = [ex.slotvalue for ex in ex_list]
        # 获取batch里各个标注的'B/I-动作-语义槽'嵌入表示列表。以最长的那个文本为最大长度，不足用PAD补齐
        tag_lens = [len(ex.tag_id) for ex in ex_list]
        max_tag_lens = max(tag_lens)
        tag_ids = [ex.tag_id + [tag_pad_idx] * (max_tag_lens - len(ex.tag_id)) for ex in ex_list]
        # 将PAD用掩码掩盖掉
        tag_mask = [[1] * len(ex.tag_id) + [0] * (max_tag_lens - len(ex.tag_id)) for ex in ex_list]
        # 把batch里统一长度的Example词标注列表转换为张量，用于训练
        batch.tag_ids = torch.tensor(tag_ids, dtype=torch.long, device=device)
        batch.tag_mask = torch.tensor(tag_mask, dtype=torch.float, device=device)
    else:
        batch.labels = None
        batch.tag_ids = None
        batch.tag_mask = None

    return batch


class Batch():

    def __init__(self, examples, device):
        super(Batch, self).__init__()

        self.examples = examples
        self.original_idx = []
        self.device = device

    def get_original_idx(self, list_idx):
        self.original_idx = list_idx

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]