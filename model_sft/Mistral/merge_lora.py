#1：merged_mistral = Mistral-7B 原始能力 + LoRA 微调后学习的 dolly_hhrlhf 5% 额外能力 
# 2：可使用 ./merged_mistral 进行 PPO 训练，并加入奖励函数，详细见ppo_training.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import shutil
import os

# 1. 设置模型路径
BASE_MODEL = "mistralai/Mistral-7B-v0.1"  # Mistral 7B 原始模型
LORA_PATH = "./finetuned_model"  # LoRA 微调后的适配层
MERGED_MODEL_PATH = "./merged_mistral"  # 最终合并的模型存放位置

# 2. 先加载 Mistral-7B 基础模型
print("正在加载 Mistral-7B 基础模型...")
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)

# 3. 加载 LoRA 适配层
print("正在加载 LoRA 适配层...")
model = PeftModel.from_pretrained(model, LORA_PATH)

# 4. 合并 LoRA，使模型变成完整的新模型
print("正在合并 LoRA 到 Mistral-7B...")
model = model.merge_and_unload()

# 5. 保存合并后的完整模型
print(f"正在保存合并后的模型到 {MERGED_MODEL_PATH}...")
model.save_pretrained(MERGED_MODEL_PATH)

# 6. 确保 `merged_mistral` 目录中包含 tokenizer
print("正在复制 tokenizer 相关文件...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(MERGED_MODEL_PATH)

print("LoRA 合并完成，完整模型已保存到:", MERGED_MODEL_PATH)

# 7. 测试能否正确加载合并后的模型和 tokenizer
print("正在测试合并后的模型是否可用...")
try:
    test_model = AutoModelForCausalLM.from_pretrained(MERGED_MODEL_PATH, torch_dtype=torch.float16)
    test_tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH)
    print("模型和 tokenizer 加载成功！")
except Exception as e:
    print("加载失败，错误信息:", str(e))

