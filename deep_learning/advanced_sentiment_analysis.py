import gradio as gr
import matplotlib.pyplot as plt
import tempfile
import os
from transformers import pipeline
import torch # 导入torch库

# 1. 定义你本地模型的路径 (./ 表示当前目录)
model_path = "./roberta-goemotions-finetuned/checkpoint-4071"

# 2. 自动检测是否有可用的GPU
#    如果有NVIDIA GPU (CUDA)，则使用GPU (device=0)，否则使用CPU (device=-1)
device = 0 if torch.cuda.is_available() else -1
print(f"使用的设备: {'GPU' if device == 0 else 'CPU'}")

# 3. 加载你本地的模型
classifier = pipeline(
    "text-classification",
    model=model_path,
    top_k=None,
    device=device
)

# Define emotion categories
# 注意：这里的标签顺序和名称需要和你训练时完全一致
# GoEmotions有27个标签 + 1个neutral。这里为了演示，可以先用主要的几个。
# 如果需要显示所有28个，可以把它们都列出来。
emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]


# Process input text and obtain emotion classification
def analyze_sentiment(text):
    # pipeline现在会返回一个包含所有标签概率的列表
    results = classifier(text)[0]
    probabilities = {res['label']: res['score'] for res in results}
    return probabilities

# Generate a probability distribution bar chart
def plot_probabilities(probabilities):
    top_k = 15
    # 按概率从高到低排序
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    if not sorted_probs:
        return None

    labels, scores = zip(*sorted_probs)

    plt.figure(figsize=(8, max(4, len(labels) * 0.4)))
    plt.barh(labels, scores, color="skyblue")
    plt.xlabel("Probability")
    plt.title("Emotion Classification Probabilities")
    plt.xlim(0, 1)
    plt.gca().invert_yaxis()

    plt.tight_layout()

    temp_dir = tempfile.gettempdir()
    image_path = os.path.join(temp_dir, "emotion_probabilities.png")
    plt.savefig(image_path)
    plt.close()
    return image_path


# Gradio interface function
def gradio_sentiment_analysis(text):
    if not text.strip():
        return "请输入文本", {}, None # 处理空输入

    probabilities = analyze_sentiment(text)
    image_path = plot_probabilities(probabilities)

    predicted_emotion = max(probabilities, key=probabilities.get)

    return predicted_emotion, probabilities, image_path

# Launch Gradio interface
iface = gr.Interface(
    fn=gradio_sentiment_analysis,
    inputs=gr.Textbox(lines=2, placeholder="Enter an English sentence..."),
    outputs=[
        gr.Textbox(label="Predicted Emotion"),
        gr.Label(label="Class Probabilities"),
        gr.Image(label="Emotion Probability Distribution")
    ],
    title="🎭 My Fine-Tuned Sentiment Analysis System",
    description="Enter an English sentence to predict its sentiment category. This system uses a RoBERTa model fine-tuned on the GoEmotions dataset.",
)

iface.launch()