#使用Medusa 对原模型Mistral-7B进行推理加速
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# from medusa import apply_medusa

MODEL_NAME = "./merged_mistral"
USE_MPS = torch.backends.mps.is_available()

def load_model():
    """加载 LLM 并优化推理"""
    device = "mps" if USE_MPS else "cpu"

    print(f"🔹 加载 {MODEL_NAME}，使用设备：{device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # model = apply_medusa(model)

    # 直接使用默认的 Transformer（不再手动转换 BetterTransformer）
    # 你的 Mistral 7B 版本 已经默认使用了 BetterTransformer 优化，所以 不需要手动调用 BetterTransformer.transform(model)。
    model.eval()

    return model, tokenizer
