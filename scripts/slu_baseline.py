#coding=utf8
import json
import sys, os, time, gc
from torch.optim import Adam
from tqdm import tqdm

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from utils.args import init_args
from utils.initialization import *
from utils.example import Example
from utils.batch import from_example_list
from utils.vocab import CLS, PAD, SEP, LabelVocab
from utils.correction import Correction
from model.slu_bert_tagging import SLUTagging

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
test_path = os.path.join(args.dataroot, 'test_unlabelled.json')
output_path = os.path.join(args.dataroot, 'test.json')
Example.configuration(args.dataroot, args.local, not args.output)
if not args.output:
    train_dataset = Example.load_dataset(train_path)
    dev_dataset = Example.load_dataset(dev_path)
    print("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
    print("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))
else:
    test_dataset = Example.load_testset(test_path)
    print("Load test dataset finished, cost %.4fs ..." % (time.time() - start_time))
    print("Dataset size: test -> %d" % len(test_dataset))

args.CLS_idx = Example.label_vocab.convert_tag_to_idx(CLS)
args.SEP_idx = Example.label_vocab.convert_tag_to_idx(SEP)
args.num_tags = Example.label_vocab.num_tags        # 标注库大小
args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)
args.num_intents = 2


model = SLUTagging(args).to(device)
corrector = Correction(args.dataroot) if args.corrector else None

def set_optimizer(model, args):
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grouped_params = [{'params': list(set([p for n, p in params]))}]
    optimizer = Adam(grouped_params, lr=args.lr)
    return optimizer


def decode(choice):
    assert choice in ['train', 'dev']
    if choice == 'dev':
        model.load_state_dict(torch.load('model.bin', map_location=device)['model'])
    model.eval()    # 设置为eval模式，固定dropout的值
    dataset = dev_dataset
    predictions, labels = [], []    # 整个数据集的预测结果、实际标注列表。都是'动作-语义槽-槽值'的可读形式
    total_slot_loss, total_intent_loss, total_joint_loss, count = 0, 0, 0, 0
    with torch.no_grad():   # decode阶段不需要计算梯度
        for i in tqdm(range(0, len(dataset), args.batch_size), desc='│ Evaluation progress'):
            # 按照batch大小，分批向模型输入数据
            cur_dataset = dataset[i: i + args.batch_size]
            current_batch = from_example_list(args, cur_dataset, device, train=True)
            # 获取当前batch的预测结果、实际标注和loss
            pred, label, slot_loss, intent_loss, joint_loss = model.decode(Example.label_vocab, current_batch, corrector)
            predictions.extend(pred)    # 将这个batch的预测结果添加到整个数据集的列表
            labels.extend(label)    # 将这个batch的实际标注添加到数据集的列表
            total_slot_loss += slot_loss
            total_intent_loss += intent_loss
            total_joint_loss += joint_loss
            count += 1
        metrics = Example.evaluator.acc(predictions, labels)    # 评价结果，使用acc和fscore
    torch.cuda.empty_cache()
    gc.collect()
    # decode返回评价结果与loss
    return metrics, total_slot_loss / count, total_intent_loss / count, total_joint_loss / count


def output():
    model_ckpt = torch.load('model.bin', map_location=device)
    model.load_state_dict(model_ckpt['model'])
    model.eval()
    dataset = test_dataset
    predictions = []
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), args.batch_size), desc='> Testing progress'):
            cur_dataset = dataset[i: i + args.batch_size]
            current_batch = from_example_list(args, cur_dataset, device, train=False) 
            pred = model.decode(Example.label_vocab, current_batch, corrector)
            predictions.extend(pred)
    
    datas = json.load(open(test_path, 'r', encoding='utf-8'))
    count = 0
    for data in datas:
        for utt in data:
            # utt['pred'] = original_pred[count]
            utt['pred'] = [pred_tuple.split('-') for pred_tuple in predictions[count]]
            count += 1
    json.dump(datas, open(output_path, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)


if not args.testing and not args.output:    # 如果不是开发集/测试集状态：（即当前处于训练状态）
    # 训练次数 = batch总数 * max_epoch
    num_training_steps = ((len(train_dataset) + args.batch_size - 1) // args.batch_size) * args.max_epoch
    print('Total training steps: %d' % (num_training_steps))
    optimizer = set_optimizer(model, args)
    nsamples, best_result = len(train_dataset), {'dev_acc': 0., 'dev_f1': 0.}
    train_index, step_size = np.arange(nsamples), args.batch_size
    print('Start training ......')
    for i in range(args.max_epoch):
        start_time = time.time()
        epoch_joint_loss = 0
        epoch_slot_loss, epoch_intent_loss = 0, 0
        epoch_manual_slot_loss = 0
        np.random.shuffle(train_index)  # 随机打乱训练集
        model.train()   # 设置为训练模式，进行梯度下降
        count = 0
        for j in tqdm(range(0, nsamples, step_size), desc='┌ Training progress'):
            # 从数据集中获取batch_size个Example的列表
            cur_dataset = [train_dataset[k] for k in train_index[j: j + step_size]]
            # 并从这个列表创建batch
            current_batch = from_example_list(args, cur_dataset, device, train=True)
            # 前向传递，获得输出的预测标注与loss
            output, slot_loss, manual_slot_loss, intent_loss, joint_loss = model(current_batch)
            epoch_joint_loss += joint_loss.item()
            epoch_slot_loss += slot_loss.item()
            epoch_manual_slot_loss += manual_slot_loss.item()
            epoch_intent_loss += intent_loss.item()
            # 反向传递，更新模型参数
            joint_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            count += 1
        print('│ Training: \tEpoch: %d\tTime: %.4f\tTraining jointLoss: %.4f (slotLoss: asr %.4f | manual %.4f, intentLoss: %.4f)' % (i, time.time() - start_time, epoch_joint_loss / count, epoch_slot_loss / count, epoch_manual_slot_loss / count, epoch_intent_loss / count))
        torch.cuda.empty_cache()
        gc.collect()
        
        # 经过一个epoch的训练后，在开发集上进行测试
        start_time = time.time()
        metrics, dev_slot_loss, dev_intent_loss, dev_joint_loss = decode('train')
        dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
        print('└ Evaluation: \tEpoch: %d\tTime: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, time.time() - start_time, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
        if dev_acc > best_result['dev_acc']:
            best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1'], best_result['iter'] = dev_joint_loss, dev_acc, dev_fscore, i
            torch.save({
                'model': model.state_dict(),
            }, open('model_best.bin', 'wb'))
            print('└ NEW BEST MODEL: \tEpoch: %d\tDev joint loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, dev_joint_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))

    print('FINAL BEST RESULT: \tEpoch: %d\tDev joint loss: %.4f\tDev acc: %.4f\tDev fscore(p/r/f): (%.4f/%.4f/%.4f)' % (best_result['iter'], best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1']['precision'], best_result['dev_f1']['recall'], best_result['dev_f1']['fscore']))
    torch.save({'epoch': i, 
                'model': model.state_dict(),
                'optim': optimizer.state_dict()
    }, open('model_final.bin', 'wb'))
if args.testing:    # 开发集状态，只进行结果评价
    start_time = time.time()
    metrics, dev_slot_loss, dev_intent_loss, dev_joint_loss = decode('dev')
    dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
    print("Evaluation costs %.2fs ; Dev joint loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)" % (time.time() - start_time, dev_joint_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
    if args.corrector:
        corrector.save_correction_history(os.path.join(args.dataroot, 'correction_history.json'))
if args.output:     # 测试集状态，只进行结果输出
    start_time = time.time()
    predictions = output()
    print("Successfully write predictions as outputs, costs %.2fs." % (time.time() - start_time))
    if args.corrector:
        corrector.save_correction_history(os.path.join(args.dataroot, 'correction_history.json'))
