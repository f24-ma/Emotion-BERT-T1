import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load a pretrained emotion model (already fine-tuned)
model_path = "bhadresh-savani/distilbert-base-uncased-emotion"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Label mapping (taken directly from this model)
id2label = model.config.id2label

# Streamlit interface
st.title("Emotion Detection App")
st.write("Enter a sentence below to detect its emotion (high-accuracy pretrained model).")

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

        # Convert numeric confidence to text
        if confidence >= 70:
            conf_level = "High confidence"
        elif confidence >= 40:
            conf_level = "Medium confidence"
        else:
            conf_level = "Low confidence"

        # Display result
        st.success(f"Predicted emotion: {emotion} ({conf_level}, {confidence:.2f}% sure)")

        # Optional: show all probabilities
        st.write("Emotion probabilities:")
        for i, label in id2label.items():
            st.write(f"{label}: {probs[0][i]*100:.2f}%")
