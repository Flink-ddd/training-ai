from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)

input_ids = tokenizer("", return_tensors="pt")["input_ids"]

print("input_ids:", input_ids)
print("dtype:", input_ids.dtype)
print("shape:", input_ids.shape)
