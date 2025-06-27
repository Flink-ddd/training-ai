
# Step 1: Install/Upgrade all packages
#!pip install --upgrade transformers datasets accelerate evaluate fsspec -q

import transformers
import datasets
from google.colab import drive
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
import os

print("--- Libraries installed, version:", transformers.__version__, "---")

# Step 2: Mount Google Drive
drive.mount('/content/drive')
print("--- Google Drive mounted ---")

# Step 3: Set up paths, load data, and prepare label info
# Use a fresh folder name for the output to avoid conflicts
output_model_dir = "/content/drive/MyDrive/MyDissertationModels/roberta-goemotions-final-run2" 
os.makedirs(output_model_dir, exist_ok=True)

# Load the original dataset
dataset = datasets.load_dataset("go_emotions", "simplified")
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Prepare label information before data mapping
labels = dataset["train"].features["labels"].feature.names
num_labels = len(labels)
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}
print("--- Label information prepared ---")


# Step 4: Define and apply the preprocessing function
def preprocess_function(examples):
    # Tokenize the text
    tokenized_inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    # Extract the single label integer from the list of labels
    tokenized_inputs["labels"] = [label[0] for label in examples["labels"]]
    return tokenized_inputs

# Apply preprocessing and remove original columns to avoid conflicts
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=['text', 'id'])
print("--- Data loaded and processed (label format corrected) ---")


# Step 5: Load the base model (using prepared label info)
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)
print("--- Base model loaded ---")


# Step 6: Define evaluation metrics and training arguments
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    return {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}

# Configure training arguments
training_args = TrainingArguments(
    output_dir=output_model_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    report_to="none",
)
print("--- Training arguments configured ---")


# Step 7: Instantiate the Trainer and start training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,  # Pass the tokenizer to save it with the model
    compute_metrics=compute_metrics,
)

print("\n****** Starting model training! ******")
trainer.train()
print("\n****** Training complete! ******")