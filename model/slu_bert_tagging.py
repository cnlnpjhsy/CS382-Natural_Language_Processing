#coding=utf8
import torch
import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF


class SLUTagging(nn.Module):

    def __init__(self, config):
        super(SLUTagging, self).__init__()
        self.config = config
        self.testing = config.testing
        self.output = config.output
        self.bert = AutoModel.from_pretrained(config.local)
        # 编码层的输出对每个字都是一层隐藏层（默认维度为768维），用于接下来的解码
        # 例如：[CLS]我要去北京[SEP]，最终输出得到(1, 7, 768)维度的张量
        self.output_layer = TaggingFNNDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx)
        self.CRF_output = TaggingCRF(config.num_tags)
        self.intent_output = IntentFNNDecoder(config.hidden_size, 2)
        self.slot_loss_weight = config.slot_loss
        self.intent_loss_weight = config.intent_loss

    def forward(self, batch):
        # 输入句子中词标注的嵌入表示，相当于输出y
        tag_ids = batch.tag_ids
        manual_tag_ids = batch.manual_tag_ids
        tag_mask = batch.tag_mask
        manual_tag_mask = batch.manual_tag_mask
        intents = batch.intents
        # 输入句子中词的bert表示，相当于输入x
        input_idx = batch.input_idx
        manual_input_idx = batch.manual_input_idx
        input_type_idx = batch.input_type_idx
        manual_input_type_idx = batch.manual_input_type_idx
        input_attn_mask = batch.input_attn_mask
        manual_input_attn_mask = batch.manual_input_attn_mask

        # 将词的bert编码送入预训练模型，得到隐藏层张量
        hiddens = self.bert(input_idx, input_type_idx, input_attn_mask).last_hidden_state
        # bert的隐藏层张量送入decoder中，返回各个标注概率
        tag_logits = self.output_layer(hiddens)
        # 标注概率再送入条件随机场，训练转移概率矩阵得到最佳标注
        CRF_tag_output = self.CRF_output(tag_logits, tag_ids, tag_mask)
        if not self.output:
            # 训练条件下，manual文本也一同送入模型
            manual_hiddens = self.bert(manual_input_idx, manual_input_type_idx, manual_input_attn_mask).last_hidden_state
            manual_tag_logits = self.output_layer(manual_hiddens)
            manual_CRF_tag_output = self.CRF_output(manual_tag_logits, manual_tag_ids, manual_tag_mask)
            # 联合训练时，将[CLS]对应hiddens输入意图预测中
            intent_output = self.intent_output(hiddens[:, :1, :], intents)

        if not self.output:
            tag_seq = CRF_tag_output[0]
            slot_loss = CRF_tag_output[1]
            manual_slot_loss = manual_CRF_tag_output[1]
            intent_loss = intent_output
            joint_loss = self.slot_loss_weight * (slot_loss * 0.50 + manual_slot_loss * 0.50) + self.intent_loss_weight * intent_loss
            return tag_seq, slot_loss, manual_slot_loss, intent_loss, joint_loss
        # 训练时，返回预测标注与loss；测试时，只返回预测标注
        return CRF_tag_output

    def decode(self, label_vocab, batch, corrector):   # 在开发集上使用decode函数
        batch_size = len(batch)
        labels = batch.labels   # 整个batch的实际标注，'动作-语义槽-槽值'的可读形式
        if not self.output:
            tag_seq, slot_loss, manual_slot_loss, intent_loss, joint_loss = self.forward(batch)    # 获取模型输出的标注与loss
        else:
            tag_seq = self.forward(batch)
        predictions = []    # 整个batch的预测结果，'动作-语义槽-槽值'的可读形式
        for i in range(batch_size):     # 对batch里的第i个Example：
            pred = tag_seq[i][1 : -1]
            pred_tuple = []     # batch中其中一个Example的预测结果
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])]     # 将PAD部分的概率截去
            for idx, tid in enumerate(pred):    # 迭代预测结果数组（迭代每一个字）：
                tag = label_vocab.convert_idx_to_tag(tid)   # 将标签的嵌入表示转回'B/I-动作-语义槽'表示
                pred_tags.append(tag)
                # 对槽值进行标注处理
                # 当前一个标注的槽值已经处理完毕时，转入该if，将buffer内存储的字idx整合为槽值
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    # 将该槽值第一个字（标注为B）的预测语义槽，作为整个槽值的语义槽
                    slot = '-'.join(tag_buff[0].split('-')[1:])     # '动作-语义槽'
                    value = ''.join([batch.utt[i][j] for j in idx_buff])    # 从idx_buff得到对应槽值
                    idx_buff, tag_buff = [], []     # 清空buffer
                    pred_tuple.append(f'{slot}-{value}')    # 添加预测结果'动作-语义槽-槽值'
                    if tag.startswith('B'):     # 如果这个字是新语义槽的开始，则继续填充buffer
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                # 否则，说明这一个槽值还没处理完毕/这是一个新的标注槽值要处理，转入该elif，继续填充buffer
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:   # 迭代结束后，处理buffer内的剩余部分
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')    # 添加预测结果'动作-语义槽-槽值'
            # 进行poi修正。训练时不修正
            if corrector is not None and (self.testing or self.output):
                pred_tuple = corrector(batch.utt[i], pred_tuple)
            predictions.append(pred_tuple)  # 将这个Example的预测结果添加到batch的预测结果中
            # 到此为止完成了batch里其中一个Example的解码过程。继续循环batch内所有的Example
        if not self.output:
            # 返回预测结果、实际标注与loss
            return predictions, labels, slot_loss.cpu().item(), intent_loss.cpu().item(), joint_loss.cpu().item()
        return predictions


class TaggingFNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)

    def forward(self, hiddens):
        logits = self.output_layer(hiddens)     # 接全连接层，得到预测输出logits
        return logits


class TaggingCRF(nn.Module):
    def __init__(self, num_tags):
        super(TaggingCRF, self).__init__()
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, logits, labels=None, mask=None):
        if labels is not None:
            tag_seq = self.crf.decode(logits, mask=mask.bool())
            neg_log_likelihood = -self.crf.forward(logits, labels, mask=mask.bool())    # loss
            return tag_seq, neg_log_likelihood
        tag_seq = self.crf.decode(logits)
        return tag_seq


class IntentFNNDecoder(nn.Module):
    def __init__(self, input_size, num_intents):
        super(IntentFNNDecoder, self).__init__()
        self.num_intents = num_intents
        self.output_layer = nn.Sequential(
            nn.Linear(input_size, num_intents),
            nn.Sigmoid()
        )
        self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, hiddens, intents=None):
        logits = self.output_layer(hiddens).squeeze()
        if intents is not None:
            intent_loss = self.loss_fct(logits, intents.float())
            return intent_loss
        return
        