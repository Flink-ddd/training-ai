import json
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from pydantic import BaseModel, Field, create_model
from typing import Literal, List, Optional

# --- 1. 初始化 LLM，保留我们用于调试的关键参数 ---
print("--- 开始初始化 VLLM 引擎 ---")
llm = LLM(
    model="google/gemma-3-27b-it",
    quantization="bitsandbytes",
    trust_remote_code=True,
    max_model_len=2048,
    max_num_seqs=100,             # <-- 采用了原作者的参数
    tensor_parallel_size=1,
    enforce_eager=True,      # <-- 这是我们能进断点的关键，必须保留！
    gpu_memory_utilization=0.8           
)
print("--- VLLM 引擎初始化完毕 ---")


# --- 2. 完全采用原作者的 Prompt 和 JSON Schema ---
TYPE = "review on airlines posted on TripAdvisor"

json_schema = {'properties': {'gender': {'enum': ['male', 'female', 'non-binary', 'other', 'prefer_not_to_say'], 'title': 'Gender', 'type': 'string'}, 'prename': {'title': 'Prename', 'type': 'string'}, 'surname': {'title': 'Surname', 'type': 'string'}, 'age': {'maximum': 120, 'minimum': 0, 'title': 'Age', 'type': 'integer'}, 'overall_mood': {'enum': ['joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust', 'anger', 'anticipation'], 'title': 'Overall Mood', 'type': 'string'}, 'persona_description': {'title': 'Persona Description', 'type': 'string'}, 'review_title': {'title': 'Review Title', 'type': 'string'}, 'travel_class': {'title': 'Travel Class', 'type': 'string'}, 'trip_type': {'title': 'Trip Type', 'type': 'string'}, 'number_of_passengers': {'title': 'Number Of Passengers', 'type': 'string'}, 'flight_distance': {'title': 'Flight Distance', 'type': 'string'}, 'reason_for_travel': {'title': 'Reason For Travel', 'type': 'string'}, 'airline_loyalty_program_member': {'title': 'Airline Loyalty Program Member', 'type': 'string'}, 'previous_airline_experience': {'title': 'Previous Airline Experience', 'type': 'string'}, 'booking_method': {'title': 'Booking Method', 'type': 'string'}, 'meal_preference': {'title': 'Meal Preference', 'type': 'string'}, 'entertainment_preference': {'title': 'Entertainment Preference', 'type': 'string'}}, 'required': ['gender', 'prename', 'surname', 'age', 'overall_mood', 'persona_description', 'review_title', 'travel_class', 'trip_type', 'number_of_passengers', 'flight_distance', 'reason_for_travel', 'airline_loyalty_program_member', 'previous_airline_experience', 'booking_method', 'meal_preference', 'entertainment_preference'], 'title': 'ReviewMeta', 'type': 'object'}


# --- 3. 完全采用原作者的采样参数 ---
guided_decoding_params = GuidedDecodingParams(json=json_schema)

sampling_params = SamplingParams(
    temperature=0.8,
    max_tokens=8192,
    seed=2,
    guided_decoding=guided_decoding_params
)

# --- 4. 准备 Prompt ---
prompt = f"Return the metadata of a person who writes a {TYPE}: "
print(f"使用的 Prompt: {prompt}")

# --- 5. 运行生成 ---
print("\n--- 开始调用 llm.generate() ---")
# 注意：原作者的 prompt 是一个列表，我们也保持一致
outputs = llm.generate([prompt], sampling_params)
print("--- llm.generate() 调用结束 ---\n")


# --- 6. 保留我们更健壮的输出和解析逻辑 ---
for output in outputs:
    generated_text = output.outputs[0].text
    print("="*30)
    print("模型原始输出 (Raw Output):")
    print("="*30)
    print(f"{generated_text!r}")
    print("="*30)
    
    print("\n尝试解析 JSON...")
    try:
        parsed_json = json.loads(generated_text)
        print("JSON 解析成功!")
        print(parsed_json)
    except json.JSONDecodeError as e:
        print(f"JSON 解析失败: {e}")
        print("这通常是因为模型在中途开始无限输出换行符，导致JSON结构不完整。")
    print("="*30)