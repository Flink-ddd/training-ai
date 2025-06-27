import gradio as gr
import matplotlib.pyplot as plt
import tempfile
import os
from transformers import pipeline
import torch # Import the torch library

# --- Main Configuration ---

# 1. Define the path to your local model folder
#    Remember to point this to the correct checkpoint folder after training
model_path = "./roberta-goemotions-finetuned/checkpoint-4071"


# 2. Automatically detect if a GPU is available
#    Use GPU (device=0) if available (NVIDIA CUDA), otherwise use CPU (device=-1)
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

# 3. Load your local model via the pipeline
classifier = pipeline(
    "text-classification",
    model=model_path,
    top_k=None,
    device=device
)

# --- End of Configuration ---


# Emotion labels from the GoEmotions dataset
emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]


# Process input text and obtain emotion classification
def analyze_sentiment(text):
    # The pipeline now returns a list containing probabilities for all labels
    results = classifier(text)[0]
    probabilities = {res['label']: res['score'] for res in results}
    return probabilities

# Generate a probability distribution bar chart
def plot_probabilities(probabilities):
    top_k = 15
    # Sort probabilities from highest to lowest
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    if not sorted_probs:
        return None

    labels, scores = zip(*sorted_probs)

    plt.figure(figsize=(8, max(4, len(labels) * 0.4)))
    plt.barh(labels, scores, color="skyblue")
    plt.xlabel("Probability")
    plt.title("Emotion Classification Probabilities")
    plt.xlim(0, 1)
    plt.gca().invert_yaxis() # Display highest probability at the top

    plt.tight_layout() # Adjust spacing to prevent overlap

    temp_dir = tempfile.gettempdir()
    image_path = os.path.join(temp_dir, "emotion_probabilities.png")
    plt.savefig(image_path)
    plt.close()
    return image_path


# Gradio interface function
def gradio_sentiment_analysis(text):
    # Handle empty input
    if not text.strip():
        return "Please enter some text", {}, None 

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
    title="🎭 My Fine-Tuned Sentiment Analysis System",
    description="Enter an English sentence to predict its sentiment category. This system uses a RoBERTa model fine-tuned on the GoEmotions dataset.",
)

iface.launch()