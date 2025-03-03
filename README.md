# NLP 新闻分类赛

## 1. 项目介绍

本项目是基于 **BERT** 进行 **新闻文本分类** 的任务，数据集由 **14 个类别的脱敏新闻文本** 组成。为了更好地适配该数据，本项目采用 **两阶段训练** 方法：

1. **预训练 BERT**：在原始脱敏语料上进行无监督预训练，以增强模型对该语料库的理解能力。
2. **微调 BERT**：在预训练好的 BERT 模型上，使用标注数据进行新闻分类任务的训练。

## 2. 赛题背景

### 2.1 赛题数据

赛题以新闻数据为赛题数据，数据集报名后可见并可下载。赛题数据为新闻文本，并按照字符级别进行匿名处理。整合划分出14个候选分类类别：财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐的文本数据。
赛题数据由以下几个部分构成：训练集20w条样本，测试集A包括5w条样本，测试集B包括5w条样本。为了预防选手人工标注测试集的情况，我们将比赛数据的文本按照字符级别进行了匿名处理。处理后的赛题训练数据如下：

| label |                             text                             |
| :---: | :----------------------------------------------------------: |
|   6   | 57 44 66 56 2 3 3 37 5 41 9 57 44 47 45 33 13 63 58 31 17 47 0 1 1 69 26 60 62 15 21 12 49 18 38 20 50 23 57 44 45 33 25 28 47 22 52 35 30 14 24 69 54 7 48 19 11 51 16 43 26 34 53 27 64 8 4 42 36 46 65 69 29 39 15 37 57 44 45 33 69 54 7 25 40 35 30 66 56 47 55 69 61 10 60 42 36 46 65 37 5 41 32 67 6 59 47 0 1 1 68 |

在数据集中标签的对应的关系如下：

```json
{'科技': 0, '股票': 1, '体育': 2, '娱乐': 3, '时政': 4, '社会': 5, '教育': 6, '财经': 7, '家居': 8, '游戏': 9, '房产': 10, '时尚': 11, '彩票': 12, '星座': 13}
```

赛题数据来源为互联网上的新闻，通过收集并匿名处理得到。因此选手可以自行进行数据分析，可以充分发挥自己的特长来完成各种特征工程，不限制使用任何外部数据和模型。
数据列使用\t进行分割，Pandas读取数据的代码如下：

```python
train_df = pd.read_csv('../input/train_set.csv', sep='\t')
```

### 2.2 任务目标

对 **匿名处理后的新闻文本** 进行分类。

### 2.3 评测标准

评价标准为类别f1_score的均值，选手提交结果与实际测试集的类别进行对比，结果越大越好。

计算公式如下：
$$
F_1=2*\frac{precision*recall}{precision+recall}
$$
可以通过sklearn完成f1_score计算：

```python
from sklearn.metrics import f1_score
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
f1_score(y_true, y_pred, average='macro')
```

### 2.4 结果提交

提交前请确保预测结果的格式与sample_submit.csv中的格式一致，以及提交文件后缀名为csv。

## 3. 环境准备

### 3.1 安装依赖

```bash
conda create -n nlp_news python=3.10 -y
conda activate nlp_news

pip install torch transformers datasets tqdm pandas scikit-learn
```

### 3.2 目录结构

```css
NLP_Competition/
├── data/                         # 数据集存放目录
│   ├── README.txt                 # 数据说明文档
│   ├── train_set.csv               # 训练集
│   ├── test_a.csv                  # 测试集 A
│   ├── test_a_sample_submit.csv     # 测试集 A 提交示例
│
├── finally_bert/                  # 训练完成的 BERT 模型
│   ├── config.json                 # BERT 配置文件
│   ├── model.safetensors            # 模型参数文件
│   ├── special_tokens_map.json      # 特殊 Token 映射
│   ├── tokenizer_config.json        # Tokenizer 配置
│   ├── tokenizer.json               # Tokenizer 文件
│   ├── training_args.bin            # 训练超参数
│
├── lm/                             # 语言模型训练相关目录
│   ├── news-classification-2/      # 训练记录存档
│   ├── events.out.tfevents...      # TensorBoard 训练日志
│
├── pre_Bert/                       # 预训练 BERT 模型
│   ├── config.json                 # 预训练模型配置
│   ├── generation_config.json       # 生成任务配置
│   ├── model.safetensors            # 预训练模型权重
│   ├── training_args.bin            # 训练参数文件
│
├── Test-Clm/                        # 训练过程中的多个检查点
│   ├── checkpoint-9000
│   ├── checkpoint-18000
│   ├── checkpoint-27000
│   ├── checkpoint-36000
│   ├── checkpoint-37074
│
├── runs/                           # 训练日志存储
│
├── tmp_trainer/                    # 训练过程中临时存储文件
│
├── 课程链接.txt                     # 相关课程链接（可能是学习资源）
├── README.md                        # 项目说明文档
├── 智慧笔记：NLP新闻分类赛.ipynb     # Jupyter Notebook 文件，包含代码和实验记录
│
├── submit_modelscope.csv            # ModelScope运行结果
├── submit0217.csv                    # 2024-02-17 提交的预测结果
├── submit1022.csv                    # 2024-10-22 提交的预测结果
```

## 4. 数据预处理

**文本数据处理流程：**

1. **加载数据**（Pandas）
2. **构建分词器**
3. **转换为 BERT 输入格式**（`input_ids`, `attention_mask`, `token_type_ids`）

代码示例：

```python
# 将3750/648/900改成标点符号，删除原text列，新增words列重命名为text列
import re
def replacepunc(x):
    x = re.sub('3750', ',', x)
    x = re.sub('900', '.', x)
    x = re.sub('648', '!', x)
    return x

df['words'] = df['text'].map(lambda x: replacepunc(x))
df.drop('text', axis=1, inplace=True)
df.columns = ['label', 'text']

# 数据载入dataset，去除多余的列，只保留text列
data = Dataset.from_pandas(df).remove_columns(['label', '__index_level_0__'])

batch_size = 100
def batch_iterator():
    for i in range(0, len(data), batch_size):
        yield data['text'][i : i + batch_size]

# 设置分词器并进行训练
# 初始化分词器、预分词器
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer

tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]

# 上面通过数据分析知道字符数量是6.9k，所以词表大小设置为7k
trainer = trainers.WordPieceTrainer(vocab_size=7000, min_frequency=2, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

tokenizer.decodes = decoders.WordPiece(prefix="##") # wordpiece的前缀

# 开始训练
tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
```

## 5. 训练与微调

### 5.1 预训练 BERT

本项目使用脱敏语料对 **BERT** 进行额外的无监督预训练，以更好地适应比赛文本。

```python
from transformers import BertConfig

# 初始化BERT模型的配置
config = BertConfig(
    vocab_size=7000,               # 词汇表大小，设置为7000，适用于自定义词表
    hidden_size=512,               # Transformer隐藏层的维度
    intermediate_size=4 * 512,      # Feed-forward网络的隐藏层大小，通常设置为 4 * hidden_size
    max_position_embeddings=512,    # 最大位置编码（支持的最大序列长度）
    num_hidden_layers=4,            # Transformer的编码层数量（BERT默认12层，这里减少为4层以降低计算量）
    num_attention_heads=4,          # 自注意力机制的多头注意力数（默认BERT Base是12头，这里降低到4头）
    type_vocab_size=2               # 标记句子对任务（如NSP）的token类型数量
)

from transformers import BertForMaskedLM

# 根据上述配置初始化BERT模型（即一个新的、未经过预训练的BERT）
model = BertForMaskedLM(config=config)

# 另一种方式：可以加载已有的预训练模型
# model = BertForMaskedLM.from_pretrained()
print(model)
```

```json
BertForMaskedLM(
  (bert): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(7000, 512, padding_idx=0)
      (position_embeddings): Embedding(512, 512)
      (token_type_embeddings): Embedding(2, 512)
      (LayerNorm): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0-3): 4 x BertLayer(
          (attention): BertAttention(
            (self): BertSdpaSelfAttention(
              (query): Linear(in_features=512, out_features=512, bias=True)
              (key): Linear(in_features=512, out_features=512, bias=True)
              (value): Linear(in_features=512, out_features=512, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=512, out_features=512, bias=True)
              (LayerNorm): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
...
      (decoder): Linear(in_features=512, out_features=7000, bias=True)
    )
  )
)
```

训练模型

```python
# 使用GPU训练，运行这段代码
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# 设定args和trainer准备训练，3000步看一次loss，9000步保存一次模型
from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    "Test-Clm",
    logging_strategy="steps",
    logging_steps=3000,
    save_strategy="steps",
    save_steps=9000,
    num_train_epochs=2,
    learning_rate=3e-4,
    per_device_train_batch_size=96,
    weight_decay=0.01)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets,
    data_collator=data_collator
)

# 训练并保存模型
trainer.train()

trainer.save_model("./pre_Bert")
```

### 5.2 微调 BERT 进行分类

在预训练 BERT 的基础上，使用 **带标签的新闻数据** 进行分类任务的微调。

```python
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained("models/pretrained_bert", num_labels=14)
```

训练代码：

```python
training_args = TrainingArguments(
    output_dir="models/finetuned_bert",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    save_strategy="epoch",
    logging_dir="logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
trainer.train()
```

## 6. 推理并生成结果

训练完成后，我们可以使用 **测试集** 评估模型效果。

```python
from transformers import BertForSequenceClassification, PreTrainedTokenizerFast
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset

# 加载tokenizer
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json",
                                         model_max_length=512,mask_token='[MASK]',pad_token='[PAD]',
                                         unk_token='[UNK]',cls_token='[CLS]',sep_token='[SEP]',
                                         padding_side='right',return_special_tokens_mask=True)

# 假设模型和tokenizer已经被保存到"./finally_bert"目录
model_path = "./finally_bert"

# 加载模型
model = BertForSequenceClassification.from_pretrained(model_path)

# 读取测试集并预处理
test_df = pd.read_csv('./data/test_a.csv', sep='\t')
test_df['texts'] = test_df['text'].map(lambda x: replacepunc(x))  # 确保replacepunc函数已经定义
test_df['summary'] = test_df['texts'].apply(lambda x: slipt2(x))  # 确保slipt2函数已经定义

# 加载到dataset并预处理
test_ds = Dataset.from_pandas(test_df).remove_columns(["texts", "text"])
tokenized_test_ds = test_ds.map(lambda examples: fast_tokenizer(examples['summary'], truncation=True, padding=True), batched=True)

# 实例化Trainer进行预测
from transformers import Trainer

trainer = Trainer(model=model)

# 进行预测
predictions = trainer.predict(tokenized_test_ds).predictions
pred = np.argmax(predictions, axis=1)

# 保存预测结果
pd.DataFrame({'label': pred}).to_csv('submit1022.csv', index=None)
```

## 7. 竞赛系统结果

提交`submission.csv`至竞赛官网系统上，最终得到成绩如下：

![1740989980627](img/1740989980627-1740989992046-2.png)

## 8. 结论与改进方向

### **当前方法优点**

- 采用 **BERT 预训练 + 微调**，利用无监督信息提升模型泛化能力。
- 使用 **Hugging Face Transformers** 库，易于扩展与迁移。

### **未来优化方向**

- **数据增强**：尝试 **EDA** (Easy Data Augmentation) 方法。
- **模型优化**：尝试 **ALBERT / RoBERTa / GPT** 等更强大的预训练模型。
- **模型架构调整**：在 **BERT** 架构上叠加 **RNN**
- **模型融合**：除了 **BERT** 之外，可以融合多个其他类型的模型，例如 **LogitRegression**、**FastText**、**RandomForest**