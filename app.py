
import streamlit as st
import pandas as pd
import requests
from transformers import pipeline
import matplotlib.pyplot as plt

st.set_page_config(page_title="Live News Sentiment", layout="wide")
st.title("🗞️ Live News Sentiment Analysis (India)")
st.markdown("Built with GNews + HuggingFace + Streamlit")

# 1️⃣ GNews API key input
api_key = st.text_input("🔑 Enter your GNews API key:")

# 2️⃣ Number of headlines to fetch
max_headlines = st.slider("Number of headlines to fetch:", 5, 50, 20)

if api_key:
    try:
        # 3️⃣ Fetch news from GNews
        url = f"https://gnews.io/api/v4/top-headlines?lang=en&country=in&max={max_headlines}&apikey={api_key}"
        res = requests.get(url)
        data = res.json()
        headlines = [article["title"] for article in data["articles"] if "title" in article]
        df = pd.DataFrame(headlines, columns=["headline"])

        if df.empty:
            st.warning("No headlines fetched. Check API key or quota.")
        else:
            # 4️⃣ HuggingFace sentiment pipeline
            sentiment_pipeline = pipeline("sentiment-analysis")

            # Apply sentiment analysis
            def get_sentiment(text):
                result = sentiment_pipeline(text[:512])[0]  # limit to 512 tokens
                return result['label'].lower()

            df["sentiment"] = df["headline"].apply(get_sentiment)

            # 5️⃣ Show dataframe
            st.subheader("Headlines with Sentiment")
            st.dataframe(df)

            # 6️⃣ Show sentiment distribution chart
            st.subheader("Sentiment Distribution")
            sentiment_counts = df["sentiment"].value_counts()
            st.bar_chart(sentiment_counts)

            # 7️⃣ Optional: show top positive and negative headlines
            st.subheader("Top Positive Headlines")
            st.write(df[df["sentiment"]=="positive"]["headline"].head(5))
            st.subheader("Top Negative Headlines")
            st.write(df[df["sentiment"]=="negative"]["headline"].head(5))

    except Exception as e:
        st.error(f"⚠️ Error: {e}")


