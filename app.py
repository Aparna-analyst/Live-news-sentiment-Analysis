import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

# ---- SETUP ----
st.set_page_config(page_title="Live News Sentiment", layout="wide")
st.title("🗞️ Live News Sentiment Analysis (India)")
st.markdown("Built with GNews API + HuggingFace Transformers + Streamlit")

# ---- HARDCODE YOUR API KEY HERE ----
api_key = "9182b91bcc242891443b3bf69c37b121"  # 👈 Replace this with your real key (keep private if public repo)

# ---- FETCH NEWS ----
try:
    url = f"https://gnews.io/api/v4/top-headlines?lang=en&country=in&max=30&apikey={api_key}"
    res = requests.get(url).json()
    headlines = [article["title"] for article in res["articles"] if "title" in article]
    df = pd.DataFrame(headlines, columns=["headline"])

    # ---- SENTIMENT ANALYSIS ----
    sentiment_pipeline = pipeline("sentiment-analysis")
    df["sentiment"] = df["headline"].apply(lambda x: sentiment_pipeline(x[:512])[0]['label'].lower())

    # ---- SHOW TABLE ----
    st.subheader("📰 Headlines with Sentiment")
    st.dataframe(df)

    # ---- SHOW BAR CHART ----
    st.subheader("📊 Sentiment Distribution")

    sentiment_counts = df["sentiment"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    sentiment_counts.plot(kind='bar', color=["green", "red", "gray"], ax=ax)
    ax.set_xlabel("Sentiment", fontsize=12)
    ax.set_ylabel("Number of Headlines", fontsize=12)
    ax.set_title("Sentiment Analysis of Latest News", fontsize=14)
    st.pyplot(fig)

except Exception as e:
    st.error(f"❌ Something went wrong: {e}")

