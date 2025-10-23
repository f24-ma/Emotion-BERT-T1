import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load lightweight model (fast and small)
model_path = "distilbert-base-uncased"  # or replace with your fine-tuned model path
model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=4)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Label mapping (must match your dataset)
id2label = {0: "joy", 1: "anger", 2: "sadness", 3: "neutral"}

# Streamlit interface
st.title("Emotion Detection App")
st.write("Enter a sentence below to detect its emotion (fast version).")

text = st.text_area("Enter text:")

if st.button("Predict Emotion"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        # Tokenize and move to device
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)

        # Convert logits to probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs).item()
        confidence = probs[0][pred].item() * 100
        emotion = id2label[pred]

        # Show result
# Convert numeric confidence to human-friendly label
if confidence >= 70:
    conf_level = "High confidence"
elif confidence >= 40:
    conf_level = "Medium confidence"
else:
    conf_level = "Low confidence"

st.success(f"Predicted emotion: {emotion} ({conf_level}, {confidence:.2f}% sure)")

        # Optional: show all probabilities
        st.write("Emotion probabilities:")
        for i, label in id2label.items():
            st.write(f"{label}: {probs[0][i]*100:.2f}%")
