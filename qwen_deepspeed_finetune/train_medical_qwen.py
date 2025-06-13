from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
import torch
import json

model_name = "Qwen/Qwen-1_5-7B"

# 加载模型和 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16)

# 加载自定义 JSONL 医疗数据
def load_medical_qa(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return Dataset.from_list(data)

dataset = load_medical_qa("medical_qa.jsonl")

# 格式化 prompt
def format_prompt(example):
    prompt = f"### 医生提问：\n{example['instruction']}\n\n### 医生回复：\n{example['output']}"
    tokenized = tokenizer(prompt, truncation=True, max_length=1024, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = dataset.map(format_prompt)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./qwen-medical-output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=100,
    save_strategy="epoch",
    fp16=True,
    deepspeed="deepspeed_config.json",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# 开始训练
trainer.train()
