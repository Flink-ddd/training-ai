import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "./merged_mistral"
USE_MPS = torch.backends.mps.is_available()
# 🔹 自动检测设备（支持 MPS、CUDA 和 CPU）
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def chain_of_thought_prompt(question, context=""):
    """通用 CoT 提示词，引导模型按照步骤推理"""
    return f"""
Let's think step by step and solve this problem in a logical manner.

**Task:** {question}
{f"**Existing Context:** {context}" if context else ""}

### **Guidelines for Thought Process**
1. **Understanding the problem**:
   - Clearly define the problem statement.
   - Identify any relevant background knowledge or key concepts.
   - If applicable, break down any sub-problems.

2. **Breaking down the solution**:
   - List the necessary steps to solve the problem.
   - Consider any alternative approaches and compare their effectiveness.
   - Ensure logical consistency between steps.

3. **Executing the solution process**:
   - Apply each step systematically.
   - Use intermediate reasoning to validate correctness.
   - Provide relevant explanations or calculations if necessary.

4. **Providing the final answer**:
   - Summarize the key findings.
   - Ensure the answer directly addresses the original question.
   - If applicable, highlight any key takeaways or implications.

**Now, let's apply this structured reasoning to solve the given task.**
"""

def generate_response(model, tokenizer, question, history=None):
    """执行推理并返回思维链格式的答案"""

    history = history or []  # 避免 NoneType 问题
    # 维持对话历史（如果有）
    context = " ".join([f"User: {msg[0]} AI: {msg[1]}" for msg in history])


    prompt = chain_of_thought_prompt(question, context)

    tokenizer.pad_token = tokenizer.eos_token  # 确保 tokenizer 终止生成

    # inputs = tokenizer(prompt, return_tensors="pt").to("mps" if USE_MPS else "cpu")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,  # 增加最大输出长度，确保完整推理
            eos_token_id=tokenizer.eos_token_id,  # 遇到终止符时停止
            repetition_penalty=1.2,  # 让模型减少重复
            temperature=0.4,  # 降低温度，提高逻辑性  降低随机性，提高逻辑性
            top_p=0.9,  # 限制采样范围
            do_sample=True  # 让回答更有变化
        )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    print(f"🔹 加载 {MODEL_NAME}，使用设备：{DEVICE}")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    question = "Why do we feel sleepy at night?"
    response = generate_response(model, tokenizer, question)
    
    print("💡 经过优化的回答：\n", response)