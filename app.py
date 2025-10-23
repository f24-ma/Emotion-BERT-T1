import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load your fine-tuned model
model_path = "bert_emotion_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define label mapping (must match your training)
label2id = {'anger': 0, 'joy': 1, 'sadness': 2, 'neutral': 3}
id2label = {v: k for k, v in label2id.items()}

# Streamlit app interface
st.title("Emotion Detection App")
st.write("Type any sentence and find out which emotion it expresses!")

text = st.text_area("Enter your text here:")

if st.button("Predict Emotion"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs).item()
        confidence = probs[0][pred].item() * 100
        emotion = id2label[pred]
        st.success(f"Predicted emotion: {emotion} ({confidence:.2f}% confident)")
