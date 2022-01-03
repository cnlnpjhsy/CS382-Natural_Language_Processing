#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class SLUTagging(nn.Module):

    def __init__(self, config):
        super(SLUTagging, self).__init__()
        self.config = config
        # 设置RNN的编码cell是何种方式。默认为LSTM
        self.cell = config.encoder_cell
        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        self.rnn = getattr(nn, self.cell)(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        # 编码层的输出为一层隐藏层（默认维度为512维），用于接下来的解码
        self.output_layer = TaggingFNNDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx)

    def forward(self, batch):
        # 输入句子中词标注的嵌入表示，相当于输出y
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        # 输入句子中词的idx，相当于输入x
        input_ids = batch.input_ids
        lengths = batch.lengths

        # 依照idx转换为词向量表示
        embed = self.word_embed(input_ids)
        # 将输入打包传入rnn
        packed_inputs = rnn_utils.pack_padded_sequence(embed, lengths, batch_first=True)
        packed_rnn_out, h_t_c_t = self.rnn(packed_inputs)  # batch大小 x batch的文本长度 x 词向量维度
        # 获得rnn的输出（输出为一层隐藏层）
        rnn_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out, batch_first=True)
        hiddens = self.dropout_layer(rnn_out)   # 对输出dropout
        tag_output = self.output_layer(hiddens, tag_mask, tag_ids)  # 解码输出。解码时需要用到标注
        # 训练时，返回预测标注与loss；测试时，只返回预测标注
        return tag_output

    def decode(self, label_vocab, batch):   # 在开发集上使用decode函数
        batch_size = len(batch)
        labels = batch.labels   # 整个batch的实际标注，'动作-语义槽-槽值'的可读形式
        prob, loss = self.forward(batch)    # 获取模型输出的概率与loss
        predictions = []    # 整个batch的预测结果，'动作-语义槽-槽值'的可读形式
        for i in range(batch_size):     # 对batch里的第i个Example：
            # 取softmax概率最大值作为预测结果
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
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
            predictions.append(pred_tuple)  # 将这个Example的预测结果添加到batch的预测结果中
            # 到此为止完成了batch里其中一个Example的解码过程。继续循环batch内所有的Example
        return predictions, labels, loss.cpu().item()   # 返回预测结果、实际标注与loss


class TaggingFNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens)     # 接全连接层，得到预测输出logits
        # 被mask时，对应logits为负无穷
        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)    # 分类问题，需softmax归一化
        if labels is not None:      # 有实际标注时（训练时），用交叉熵函数计算loss
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss       # 并返回标注分类概率与loss
        return prob                 # 否则没有实际标注（预测时），只返回概率
