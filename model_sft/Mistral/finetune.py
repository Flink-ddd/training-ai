# Lora微调，不改变原有模型能力，在原有模型能力上再加 mosaicml/dolly_hhrlhf 5%数据集的能力。具体优势见：Lora优点.png
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

MODEL_NAME = "mistralai/Mistral-7B-v0.1"

def fine_tune():
    """使用 LoRA 进行微调"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token  # 设置 pad_token，避免 padding 报错

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="mps"
    )

    # LoRA 配置
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8, lora_alpha=32, lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    
    model = get_peft_model(model, peft_config)

    # 只用 5% 数据集，减少训练时间，如果要用全部的话，去掉split="train[:1%]"
    dataset = load_dataset("mosaicml/dolly_hhrlhf", split="train[:5%]")

    # 处理数据（转换 `prompt` + `response` 为输入和 `labels`）
    def tokenize_function(examples):
        inputs = [f"User: {q}\nAssistant: {a}" for q, a in zip(examples["prompt"], examples["response"])]
        model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)

        # `labels` = `input_ids`
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["prompt", "response"])

    training_args = TrainingArguments(
        output_dir="./model_output",
        per_device_train_batch_size=1,  # Mac M1 只能支持 batch_size=1
        gradient_accumulation_steps=8,  # 通过梯度累积提高训练效率
        num_train_epochs=1,  # num_train_epochs 表示训练轮数（Epochs），数值 越大，训练的轮数越多，模型会在数据上学习得更充分，但同时也会 增加训练时间和计算成本。先训练 1 轮测试
        save_strategy="epoch",
        remove_unused_columns=False
    )

    model.train()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets
    )

    trainer.train()
    model.save_pretrained("./finetuned_model")

if __name__ == "__main__":
    fine_tune()
