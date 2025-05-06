import os
import praw
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import torch
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# 🌐 Suppress symlink warning from HuggingFace
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 🔄 Load emotion model
model_name = "cardiffnlp/twitter-roberta-base-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
labels = ['anger', 'joy', 'optimism', 'sadness', 'fear', 'surprise', 'disgust', 'trust']

# 🎨 Custom CSS Styling
st.markdown("""
    <style>
        .title {
            font-size: 32px;
            font-weight: bold;
            color: #4CAF50;
        }
        .subtitle {
            font-size: 18px;
            color: #555;
        }
        .post-box {
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# 🔧 Sidebar Inputs
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/5/58/Reddit_logo_new.svg", width=120)
st.sidebar.title("🔧 Settings")
subreddit_name = st.sidebar.text_input("Subreddit name", "depression")
limit = st.sidebar.slider("Number of posts", 10, 100, 50)

# 🧠 App Title
st.markdown('<div class="title">Reddit Emotion Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze emotions from Reddit posts using RoBERTa</div>', unsafe_allow_html=True)
st.write("---")

# 🛠 Connect to Reddit API
reddit = praw.Reddit(
    client_id="mDLkHdRT5fIXR2Im6igHlQ",
    client_secret="Bbl7PFz-iXO6nfNP7sAx-U2EXtXVng",
    user_agent="MySentimentApp by u/jrterex",
    username="jrterex",
    password="moon@007"
)

# 🔍 Fetch and analyze posts
with st.spinner("🔎 Analyzing posts..."):
    posts = reddit.subreddit(subreddit_name).hot(limit=limit)
    data = []

    for post in posts:
        text = post.title
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        emotion = labels[predicted_class]

        st.markdown(f"""
            <div class="post-box">
                <strong>📝 Text:</strong> {text}<br>
                <strong>❤️ Emotion:</strong> {emotion}
            </div>
        """, unsafe_allow_html=True)

        data.append({"Text": text, "Emotion": emotion})

# 💾 Save to CSV
df = pd.DataFrame(data)
csv_file = "reddit_emotions.csv"

with st.expander("📁 Download Results"):
    st.download_button(
        label="📄 Download CSV",
        data=df.to_csv(index=False),
        file_name=csv_file,
        mime="text/csv"
    )

# 📊 Emotion Chart
emotion_counts = df['Emotion'].value_counts()

st.subheader("📊 Emotion Distribution")
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
ax.set_title(f"Emotions in r/{subreddit_name}")
st.pyplot(fig)
