import streamlit as st
import requests
import pandas as pd
from transformers import pipeline

st.set_page_config(page_title="Live News Sentiment", layout="wide")

st.title("🗞️ Live News Sentiment Analysis (India)")
st.markdown("Built with GNews + HuggingFace + Streamlit")

# User input for API key (or hardcode your key for testing)
api_key = st.secrets["gnews_api"] if "gnews_api" in st.secrets else st.text_input("🔑 Enter your GNews API key")

if api_key:
    try:
        url = f"https://gnews.io/api/v4/top-headlines?lang=en&country=in&max=20&apikey={api_key}"
        res = requests.get(url).json()
        headlines = [article["title"] for article in res["articles"] if "title" in article]
        df = pd.DataFrame(headlines, columns=["headline"])

        sentiment_pipeline = pipeline("sentiment-analysis")

        df["sentiment"] = df["headline"].apply(lambda x: sentiment_pipeline(x[:512])[0]['label'].lower())

        st.success("✅ Fetched and analyzed successfully!")
        st.dataframe(df)

        # Show sentiment counts
        st.bar_chart(df["sentiment"].value_counts())

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
