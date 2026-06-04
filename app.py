import pandas as pd
import nltk
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Download NLTK data
nltk.download('punkt')

# Load CSV file
df = pd.read_csv("tweets.csv")

# Function to analyze sentiment
def analyze_sentiment(tweet):
    analysis = TextBlob(tweet)
    polarity = analysis.sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
df["Sentiment"] = df["tweet"].apply(analyze_sentiment)

# Print output
print("\n===== Sentiment Analysis Results =====\n")
print(df)

# Count sentiments
sentiment_counts = df["Sentiment"].value_counts()

# Bar chart
plt.figure(figsize=(6, 4))
sentiment_counts.plot(kind='bar')

plt.title("Twitter Sentiment Analysis")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")

plt.show()

# Word Cloud
all_tweets = " ".join(df["tweet"])

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(all_tweets)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Tweet Word Cloud")

plt.show()