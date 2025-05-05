import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

# Load model and tokenizer once (cached)
@st.cache_resource
def load_model():
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# App title
st.title("🔍 Emotion Detector from Text")
st.write("Enter a sentence below to analyze its emotion.")

# Text input
user_input = st.text_input("Your sentence here:", "")

if user_input:
    # Tokenize and run through model
    inputs = tokenizer(user_input, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1).squeeze()

    # Get emotion labels
    labels = model.config.id2label
    emotion_scores = {labels[i]: float(probs[i]) for i in range(len(probs))}

    # Sort and display
    sorted_emotions = dict(sorted(emotion_scores.items(), key=lambda item: item[1], reverse=True))
    top_emotion = max(sorted_emotions, key=sorted_emotions.get)

    st.markdown(f"### 🧠 Predicted Emotion: **{top_emotion}**")

    # Show bar chart
    st.bar_chart(pd.Series(sorted_emotions))

    # Optional: raw scores
    with st.expander("See full emotion scores"):
        st.write(sorted_emotions)
