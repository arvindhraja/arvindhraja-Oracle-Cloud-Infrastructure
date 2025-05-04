import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
import nltk
nltk.download('punkt')

# Sample Data (replace with actual tweets/comments)
data = {
    'username': ['user1', 'user2', 'user3', 'user4'],
    'text': [
        "I'm so happy today! This is amazing!",
        "I feel so sad and alone.",
        "This is frustrating and makes me angry.",
        "It's a beautiful day but I feel anxious."
    ]
}

# Load into DataFrame
df = pd.DataFrame(data)

# Function to analyze emotion
def get_emotion(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0.3:
        return 'Happy'
    elif polarity < -0.3:
        return 'Sad'
    elif -0.3 <= polarity <= 0.3:
        return 'Neutral'
    else:
        return 'Unknown'

# Apply to DataFrame
df['emotion'] = df['text'].apply(get_emotion)
df['polarity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Display DataFrame
print(df)

# Plotting
sns.countplot(data=df, x='emotion', palette='Set2')
plt.title("Detected Emotions in Social Media Texts")
plt.show()
