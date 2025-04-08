# 1：如果只是测试本地原模型：Mistral-7B的回答效果，直接运行这个文件即可。然后会加载model_loader.py和思维链的inference.py文件，进行回答输出。
# 2：如果经过Lora微调和PPO训练后的回答效果，只需要在思维链文件inference.py改MODEL_NAME即可，比如微调后的模型名是：./merged_mistral，PPO训练后的模型是：./ppo_optimized_model。

import gradio as gr
from inference import generate_response
from model_loader import load_model

# 加载模型
model, tokenizer = load_model()

# 定义对话函数
def chat(question):
    response = generate_response(model, tokenizer, question)
    return response

# 创建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 AI Chatbot - 基于你的微调模型")
    gr.Markdown("输入你的问题，模型会给出答案")

    with gr.Row():
        input_text = gr.Textbox(label="你的问题", placeholder="请输入问题...")
        output_text = gr.Textbox(label="模型回答", interactive=True)

    submit_button = gr.Button("发送")

    # 绑定交互逻辑
    submit_button.click(fn=chat, inputs=[input_text], outputs=[output_text])

# 运行 Gradio Web 界面
if __name__ == "__main__":
    demo.launch(share=True)
