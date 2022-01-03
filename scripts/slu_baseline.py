#coding=utf8
import sys, os, time, gc
from torch.optim import Adam

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from utils.args import init_args
from utils.initialization import *
from utils.example import Example
from utils.batch import from_example_list
from utils.vocab import PAD
from model.slu_baseline_tagging import SLUTagging

# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:])
set_random_seed(args.seed)
device = set_torch_device(args.device)
print("Initialization finished ...")
print("Random seed is set to %d" % (args.seed))
print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

start_time = time.time()
train_path = os.path.join(args.dataroot, 'train.json')
dev_path = os.path.join(args.dataroot, 'development.json')
Example.configuration(args.dataroot, train_path=train_path, word2vec_path=args.word2vec_path)
train_dataset = Example.load_dataset(train_path)
dev_dataset = Example.load_dataset(dev_path)
print("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
print("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))

args.vocab_size = Example.word_vocab.vocab_size     # 词库大小
args.pad_idx = Example.word_vocab[PAD]
args.num_tags = Example.label_vocab.num_tags        # 标注库大小
args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)


model = SLUTagging(args).to(device)
# 从word2vec词向量文件中，读取数据集词库的词向量表示
Example.word2vec.load_embeddings(model.word_embed, Example.word_vocab, device=device)


def set_optimizer(model, args):
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grouped_params = [{'params': list(set([p for n, p in params]))}]
    optimizer = Adam(grouped_params, lr=args.lr)
    return optimizer


def decode(choice):
    assert choice in ['train', 'dev']
    model.eval()    # 设置为eval模式，固定dropout的值
    dataset = train_dataset if choice == 'train' else dev_dataset
    predictions, labels = [], []    # 整个数据集的预测结果、实际标注列表。都是'动作-语义槽-槽值'的可读形式
    total_loss, count = 0, 0
    with torch.no_grad():   # decode阶段不需要计算梯度
        for i in range(0, len(dataset), args.batch_size):
            # 按照batch大小，分批向模型输入数据
            cur_dataset = dataset[i: i + args.batch_size]
            current_batch = from_example_list(args, cur_dataset, device, train=True)
            # 获取当前batch的预测结果、实际标注和loss
            pred, label, loss = model.decode(Example.label_vocab, current_batch)
            predictions.extend(pred)    # 将这个batch的预测结果...
            labels.extend(label)        # ...和实际标注添加到整个数据集的列表
            total_loss += loss
            count += 1
        metrics = Example.evaluator.acc(predictions, labels)    # 评价结果，使用acc和fscore
    torch.cuda.empty_cache()
    gc.collect()
    # decode返回评价结果与loss
    return metrics, total_loss / count


if not args.testing:    # 如果不是开发集/测试集状态：（即当前处于训练状态）
    # 训练次数 = batch总数 * max_epoch
    num_training_steps = ((len(train_dataset) + args.batch_size - 1) // args.batch_size) * args.max_epoch
    print('Total training steps: %d' % (num_training_steps))
    optimizer = set_optimizer(model, args)
    nsamples, best_result = len(train_dataset), {'dev_acc': 0., 'dev_f1': 0.}
    train_index, step_size = np.arange(nsamples), args.batch_size
    print('Start training ......')
    for i in range(args.max_epoch):
        start_time = time.time()
        epoch_loss = 0
        np.random.shuffle(train_index)  # 随机打乱训练集
        model.train()   # 设置为训练模式，进行梯度下降
        count = 0
        for j in range(0, nsamples, step_size):
            # 从数据集中获取batch_size个Example的列表
            cur_dataset = [train_dataset[k] for k in train_index[j: j + step_size]]
            # 并从这个列表创建batch
            current_batch = from_example_list(args, cur_dataset, device, train=True)
            # 前向传递，获得输出的预测标注与loss
            output, loss = model(current_batch)
            epoch_loss += loss.item()
            # 反向传递，更新模型参数
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            count += 1
        print('Training: \tEpoch: %d\tTime: %.4f\tTraining Loss: %.4f' % (i, time.time() - start_time, epoch_loss / count))
        torch.cuda.empty_cache()
        gc.collect()
        
        # 经过一个epoch的训练后，在开发集上进行测试
        start_time = time.time()
        metrics, dev_loss = decode('dev')
        dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
        print('Evaluation: \tEpoch: %d\tTime: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, time.time() - start_time, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
        if dev_acc > best_result['dev_acc']:
            best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1'], best_result['iter'] = dev_loss, dev_acc, dev_fscore, i
            torch.save({
                'epoch': i, 'model': model.state_dict(),
                'optim': optimizer.state_dict(),
            }, open('model.bin', 'wb'))
            print('NEW BEST MODEL: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))

    print('FINAL BEST RESULT: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.4f\tDev fscore(p/r/f): (%.4f/%.4f/%.4f)' % (best_result['iter'], best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1']['precision'], best_result['dev_f1']['recall'], best_result['dev_f1']['fscore']))
else:   # 开发集/测试集状态，只进行结果评价
    start_time = time.time()
    metrics, dev_loss = decode('dev')
    dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
    print("Evaluation costs %.2fs ; Dev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)" % (time.time() - start_time, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
