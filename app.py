import os
import praw
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import torch
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Set environment variable to suppress warning for symlinks
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load the emotion model
model_name = "cardiffnlp/twitter-roberta-base-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Map of emotions based on model config
labels = ['anger', 'joy', 'optimism', 'sadness', 'fear', 'surprise', 'disgust', 'trust']

# Streamlit UI elements
st.title("Emotion Detection from Reddit Posts")
st.write("""
    This app fetches posts from Reddit and performs emotion detection using a pre-trained RoBERTa model.
    It classifies emotions into categories like anger, joy, sadness, etc., and displays the results.
""")

# Fetch subreddit input from user
subreddit_name = st.text_input("Enter subreddit name (e.g., 'depression', 'freefire'):", "depression")

# Connect to Reddit
reddit = praw.Reddit(
    client_id="mDLkHdRT5fIXR2Im6igHlQ",
    client_secret="Bbl7PFz-iXO6nfNP7sAx-U2EXtXVng",
    user_agent="MySentimentApp by u/jrterex",
    username="jrterex",
    password="moon@007"
)

# Fetch posts from the subreddit
posts = reddit.subreddit(subreddit_name).hot(limit=100)

# 🔄 Store results for CSV
data = []

st.write("🔍 Analyzing emotions from posts...")

for post in posts:
    text = post.title

    # Tokenize and predict
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = softmax(logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()
    emotion = labels[predicted_class]

    # Display the result in the Streamlit app
    st.write(f"📝 **Text**: {text}")
    st.write(f"❤️ **Emotion**: {emotion}\n")

    # Add to data list
    data.append({"Text": text, "Emotion": emotion})

# Save results to CSV
df = pd.DataFrame(data)
csv_file = "reddit_emotions.csv"
df.to_csv(csv_file, index=False)

# Provide download button for the CSV file
st.download_button(
    label="Download results as CSV",
    data=df.to_csv(index=False),
    file_name=csv_file,
    mime="text/csv"
)

# 📊 Create and show pie chart
emotion_counts = df['Emotion'].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', colors=plt.cm.Paired.colors)
plt.title(f"Emotion Distribution from r/{subreddit_name} Posts")
plt.axis("equal")
plt.tight_layout()

# Show the pie chart in Streamlit
st.pyplot(plt)
