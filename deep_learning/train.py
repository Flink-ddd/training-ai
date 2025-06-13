
# 1: 安装/升级所有包
#!pip install --upgrade transformers datasets accelerate evaluate fsspec -q
import transformers
import datasets
from google.colab import drive
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
import os

print("--- 库已安装，版本:", transformers.__version__, "---")


# 2: 挂载 Drive
drive.mount('/content/drive')
print("--- 已挂载 Google Drive ---")


# 3: 设置路径、加载数据、并提前准备好标签信息
output_model_dir = "/content/drive/MyDrive/MyDissertationModels/roberta-goemotions-finetuned"
os.makedirs(output_model_dir, exist_ok=True)

# 加载原始数据集
dataset = datasets.load_dataset("go_emotions", "simplified")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# 在数据处理前，先从原始数据集中获取标签信息
labels = dataset["train"].features["labels"].feature.names
num_labels = len(labels)
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}
print("--- 已提前准备好标签信息 ---")


# 4: 定义预处理函数并应用它
def preprocess_function(examples):
    # 对文本进行tokenize
    tokenized_inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    # 从标签列表中提取出单个标签数字
    tokenized_inputs["labels"] = [label[0] for label in examples["labels"]]
    return tokenized_inputs
 # remove_columns以避免后续冲突
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=['text', 'id'])
print("--- 数据已加载和处理 (已修正标签格式) ---")


# 5: 加载模型 (使用提前准备好的标签信息)
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)
print("--- 模型已加载 ---")


# 6: 定义评估指标和训练参数
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    return {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}

# 配置训练参数
training_args = TrainingArguments(
    output_dir=output_model_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    report_to="none",
)
print("--- 训练参数已配置 ---")


# 7: 实例化 Trainer 并开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("\n****** 即将开始模型训练！******")
trainer.train()

print("\n****** 训练完成！******")