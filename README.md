# 大作业三：口语语义理解任务

## 更新记录

### **2021.12.15**
Repo创建，上传项目原文件。修改了几处bug，现在可以正常运行`slu_baseline.py`了。训练10个epoch，测试效果：
```
C:\Users\hanse\Desktop\课件\大三上\自然语言处理\大作业3>python scripts/slu_baseline.py --device 0 --max_epoch 10
Use GPU with index 0
Initialization finished ...
Random seed is set to 999
Use GPU with index 0
Load dataset and database finished, cost 4.1936s ...
Dataset size: train -> 5093 ; dev -> 921
Total training steps: 1600
Start training ......
Training:       Epoch: 0        Time: 5.4638    Training Loss: 0.9477
Evaluation:     Epoch: 0        Time: 0.3037    Dev acc: 71.23  Dev fscore(p/r/f): (74.09/71.55/72.80)
NEW BEST MODEL:         Epoch: 0        Dev loss: 0.5092        Dev acc: 71.23  Dev fscore(p/r/f): (74.09/71.55/72.80)
...
Training:       Epoch: 9        Time: 4.9588    Training Loss: 0.0733
Evaluation:     Epoch: 9        Time: 0.2942    Dev acc: 71.66  Dev fscore(p/r/f): (79.14/75.08/77.06)
FINAL BEST RESULT:      Epoch: 3        Dev loss: 0.4687        Dev acc: 73.0727        Dev fscore(p/r/f): (79.3527/76.0428/77.6625)
```

### **2022.1.4**  
  - 完成了对原有代码的详细注释，方便理解。
  - 在`slu_baseine.py`等文件补充了输出相关代码，现在可以输出符合要求的结果了，在运行指令中加入`--output`选项即可选择输出模式。该模式下的代码将读取`data/test_unlabelled.json`中的内容作为输入，并将结果输出至`data/test.json`。  
  可以在输入的文件中自定义一些内容，方便之后的调试与效果检验。
  ```
  python scripts/slu_baseline.py --output
  ```  
  - 从输出文件中就可以看出baseline的效果很不好。快来改进吧！
  
### **2022.1.6**
  - 将模型更换为BERT-CRF，并进行槽位填充与多意图分析的联合训练。  
    CRF层使用python库：`pytorch-crf`，使用前先pip install。  
    相关文献：
    - [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/pdf/1902.10909.pdf)
    - [Joint Multiple Intent Detection and Slot Labeling for Goal-Oriented Dialog](https://aclanthology.org/N19-1055.pdf)
  - 运行该指令进行训练。需要进行充分的训练，请调整max_epoch，并根据自身显卡状况调整batch大小。似乎每1GB显存可以支持12组数据在batch里。  
    **bert的训练很慢，请尽快进行工作，为调试与最后的fine-tuning争取时间！**
  ```
  python scripts/slu_baseline.py --max_epoch 10 --device 0 --batch_size 64
  ```

### **2022.1.14**
`model_bert.bin`
```
FINAL BEST RESULT:      Epoch: 98       Dev joint loss: 4.2948  Dev acc: 78.0673        Dev fscore(p/r/f): (86.1239/80.3209/83.1212)
```