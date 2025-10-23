import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load your fine-tuned model (change path if you have your own model)
model_path = "bert-base-uncased"  # or "yourusername/bert-emotion-model"
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=4)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Label mapping (must match your training)
id2label = {0: "joy", 1: "anger", 2: "sadness", 3: "neutral"}

# Streamlit app interface
st.title("Emotion Detection App")
st.write("Type any sentence below to find out which emotion it expresses.")

text = st.text_area("Enter your text here:")

if st.button("Predict Emotion"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

        # Model prediction
        with torch.no_grad():
            outputs = model(**inputs)

        # Convert logits to probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs).item()
        confidence = probs[0][pred].item() * 100
        emotion = id2label[pred]

        # Display result
        st.success(f"Predicted emotion: {emotion} ({confidence:.2f}% confident)")

        # Show all emotion probabilities
        st.write("Emotion probabilities:")
        for i, label in id2label.items():
            st.write(f"{label}: {probs[0][i]*100:.2f}%")
