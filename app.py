
%%writefile app.py
import streamlit as st
import pandas as pd
import requests
from transformers import pipeline
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

st.set_page_config(page_title="Live News Sentiment", layout="wide")
st.title("🗞️ Live News Sentiment Analysis (India)")
st.markdown("Built with GNews + HuggingFace + Streamlit")

# 1️⃣ API key stored in Streamlit secrets
api_key = st.secrets["gnews_api"]

# 2️⃣ Date filter: user selects date
default_date = datetime.utcnow() - timedelta(days=1)
selected_date = st.date_input("Show news after:", default_date)
from_date = datetime.combine(selected_date, datetime.min.time()).strftime("%Y-%m-%dT%H:%M:%SZ")

# 3️⃣ Number of headlines
max_headlines = st.slider("Number of headlines to fetch:", 5, 50, 20)

try:
    # 4️⃣ Fetch news from GNews API
    url = f"https://gnews.io/api/v4/top-headlines?lang=en&country=in&from={from_date}&max={max_headlines}&apikey={api_key}"
    res = requests.get(url)
    data = res.json()
    headlines = [article["title"] for article in data.get("articles", []) if "title" in article]
    
    df = pd.DataFrame(headlines, columns=["headline"])
    
    if df.empty:
        st.warning("No headlines found for the selected date.")
    else:
        # 5️⃣ HuggingFace sentiment pipeline
        sentiment_pipeline = pipeline("sentiment-analysis")

        # Apply sentiment analysis
        def get_sentiment(text):
            result = sentiment_pipeline(text[:512])[0]  # limit to 512 tokens
            return result['label'].lower()

        df["sentiment"] = df["headline"].apply(get_sentiment)

        # 6️⃣ Display dataframe
        st.subheader("Headlines with Sentiment")
        st.dataframe(df)

        # 7️⃣ Sentiment distribution chart
        st.subheader("Sentiment Distribution")
        sentiment_counts = df["sentiment"].value_counts()
        st.bar_chart(sentiment_counts)

        # 8️⃣ Optional: top positive and negative headlines
        st.subheader("Top Positive Headlines")
        st.write(df[df["sentiment"]=="positive"]["headline"].head(5))
        st.subheader("Top Negative Headlines")
        st.write(df[df["sentiment"]=="negative"]["headline"].head(5))

except Exception as e:
    st.error(f"⚠️ Error fetching news: {e}")



