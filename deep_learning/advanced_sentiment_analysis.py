import gradio as gr
import matplotlib.pyplot as plt
import tempfile
import os
from transformers import pipeline

# Load the BERT (RoBERTa) sentiment classification model
classifier = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

# Define emotion categories
emotion_labels = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]

# Process input text and obtain emotion classification
def analyze_sentiment(text):
    results = classifier(text)[0]  # Retrieve probabilities for all emotion categories
    probabilities = {res['label']: res['score'] for res in results}

    # Ensure all categories have probabilities, even if the model does not return them
    for label in emotion_labels:
        if label not in probabilities:
            probabilities[label] = 0.0

    return probabilities

# Generate a probability distribution bar chart
def plot_probabilities(probabilities):
    top_k = 15
    labels, scores = zip(*sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:top_k])

    plt.figure(figsize=(8, max(4, len(labels) * 0.4)))  # 根据类别数量动态调节图高度
    plt.barh(labels, scores, color="skyblue")
    plt.xlabel("Probability")
    plt.title("Emotion Classification Probabilities")
    plt.xlim(0, 1)
    plt.gca().invert_yaxis()  # 高概率在上面

    plt.tight_layout()  # 自动调整间距，避免重叠

    temp_dir = tempfile.gettempdir()
    image_path = os.path.join(temp_dir, "emotion_probabilities.png")
    plt.savefig(image_path)
    plt.close()
    return image_path


# Gradio interface function
def gradio_sentiment_analysis(text):
    probabilities = analyze_sentiment(text)
    image_path = plot_probabilities(probabilities)

    # Determine the predicted emotion with the highest probability
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
    title="🎭 Multi-Category Sentiment Analysis System",
    description="Enter an English sentence to predict its sentiment category. The system will return probabilities for all emotion categories.",
)

iface.launch()
