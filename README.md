# 大作业三：口语语义理解任务


## 目录结构
```
root
│  model.bin
│  README.md (This file)
│
├─data
│  │  correction_history.json
│  │  development.json
│  │  ontology.json
│  │  poi_name_ngram.json
│  │  test.json
│  │  test_statistic.json
│  │  test_unlabelled.json
│  │  train.json
│  │
│  └─lexicon
│          operation_verb.txt
│          ordinal_number.txt
│          poi_name.txt
│
├─model
│      slu_bert_tagging.py
│
├─scripts
│      slu_baseline.py
│
└─utils
       args.py
       batch.py
       correction.py
       evaluator.py
       example.py
       initialization.py
       pinyin_ngram.py
       statistic.py
       vocab.py
```

## 安装依赖
```
pip install transformers pytorch-crf Levenshtein pypinyin dimsim
```

## 运行

模型共有训练、评估、输出3种模式。

- **训练** 在`data/train.json`上进行模型训练，生成新模型。生成的最佳模型为`model_best.bin`，训练结束时的完整中间模型为`model_final.bin`。  
  在根目录下运行：`python scripts/slu_baseline.py`  
  可选参数：
  - `--max_epoch <value>`  
    模型训练的最大epoch数。默认为100。
  - `--lr <value>`  
    模型的学习率。默认为1e-5。
  - `--hidden_size <value>`  
    解码层解码时输入的隐状态层维度。默认为768（roberta的隐状态层维度）。**除非更换模型，否则不建议更改！**
  - `slot_loss <value>`  
    联合训练中，槽位填充的loss的权重。默认为0.01。
  - `intent_loss <value>`  
    联合训练中，意图识别的loss的权重。默认为1。

- **评估** 用已经训练好的模型`model.bin`在开发集`data/development.json`上进行评估。  
  在根目录下运行：`python scripts/slu_baseline.py --testing`  
  可选参数：
  - `--corrector`  
    启用地名纠错。

- **输出** 用已经训练好的模型`model.bin`读取无标注的测试文件`data/test_unlabelled.json`，并输出预测标注文件`data/test.json`。  
  在根目录下运行：`python scripts/slu_baseline.py --output`  
  可选参数：
  - `--corrector`  
    启用地名纠错。

- **其他通用参数**  
  - `--local <path>`  
    选择使用本地的roberta模型。参数值为本地模型的路径。
  - `--batch_size <value>`  
    输入到模型的batch的大小。默认为32。
  - `--dataroot <path>`  
    数据集的目录。默认为`/data`。
  - `--seed <value>`  
    随机种子值。默认为999。
  - `--device <device>`  
    使用的设备。默认为-1（CPU）。

## 结果与复现

我们的模型最终在开发集上取得的成绩如下：

| Accuracy | Precision | Recall | F1 |
| -------- | --------- | ------ | -- |
| **81.43** | **86.68** | **84.92** | **85.79** |

```
Evaluation costs 414.63s ; Dev joint loss: 1.9398       Dev acc: 81.43  Dev fscore(p/r/f): (86.68/84.92/85.79)
```
要复现结果，你可以在根目录下执行下述指令：
```
python scripts/slu_baseline.py --testing --corrector
```

## 其他实用工具

- `utils/correction.py`  
  我们的纠错程序，能够依据已有的地名数据库，对可能错误的地名槽值进行修正。该文件中的`Correction`类在程序中被调用，无需另外运行。  
  纠错程序的运行需要n-grams拼音数据`data/poi_name_ngram.json`，这是由`utils/pinyin_ngram.py`得到的。我们已经提供了一份n-grams拼音数据。

- `utils/pinyin_ngram.py`  
  为纠错程序提供n-grams拼音数据。该程序读取`data/lexicon/poi_name.txt`，为其中的地名生成n-grams拼音，并储存成.json文件`data/poi_name_ngram.json`，供纠错程序读取。  
  需要注意的是该程序并不能很好地处理含英文地名的情况（例如ktv三个字会被视作一整个拼音，导致n-grams长度出错）。代码文件中已经有一份基于当前的地名数据库获得的、且对英文内容修正过的`data/poi_name_ngram.json`了，因此不建议再运行该程序，否则可能会在纠错过程中出现错误。

- `utils/statistic.py`  
  统计有正确标注的预测输出文件`data/test.json`中正确预测与错误预测的语句，分类后储存在`data/test_statistic.json`中。可以用它来查看哪些对话被正确或者错误预测了。  
  由于要求`data/test.json`是有正确标注的文件，因此正确的使用方法应当是：先在`scripts/slu_baseline.py`中更改输出模式下的输入文件路径`test_path`为`development.json`，再运行该程序。
