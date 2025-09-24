import streamlit as st
import pandas as pd
import requests
from transformers import pipeline
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, when
from pyspark.sql.types import IntegerType

# 1️⃣ Spark Session
spark = SparkSession.builder.appName("LiveNewsSentiment").getOrCreate()

st.set_page_config(page_title="Live News Sentiment", layout="wide")
st.title("🗞️ Live News Sentiment Analysis (India)")

# 2️⃣ API Key input
api_key = st.text_input("Enter your GNews API Key:")

if api_key:
    try:
        # 3️⃣ Fetch news
        url = f"https://gnews.io/api/v4/top-headlines?lang=en&country=in&max=50&apikey={api_key}
        data = requests.get(url).json()
        headlines = [article["title"] for article in data["articles"] if "title" in article]
        df = pd.DataFrame(headlines, columns=["headline"])
        news_df = spark.createDataFrame(df)

        # 4️⃣ HuggingFace pre-trained model for pseudo-labels
        hf_model = pipeline("sentiment-analysis")
        def hf_to_label(text):
            result = hf_model(text[:512])[0]
            return 1 if result['label'].lower() == 'positive' else 0
        hf_label_udf = udf(hf_to_label, IntegerType())
        news_df = news_df.withColumn("label", hf_label_udf(news_df["headline"]))

        # 5️⃣ PySpark ML pipeline
        from pyspark.ml import Pipeline
        from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer
        from pyspark.ml.classification import LogisticRegression

        tokenizer = Tokenizer(inputCol="headline", outputCol="words")
        stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered")
        vectorizer = CountVectorizer(inputCol="filtered", outputCol="features")
        lr = LogisticRegression(featuresCol="features", labelCol="label")

        pipeline = Pipeline(stages=[tokenizer, stopwords_remover, vectorizer, lr])
        model = pipeline.fit(news_df)

        predictions = model.transform(news_df)
        predictions = predictions.withColumn("sentiment", when(col("prediction")==1, "positive").otherwise("negative"))
        pred_df = predictions.select("headline", "sentiment").toPandas()

        # 6️⃣ Display in Streamlit
        st.success("✅ News fetched and analyzed!")
        st.dataframe(pred_df)

        # 7️⃣ Sentiment bar chart
        st.bar_chart(pred_df['sentiment'].value_counts())

    except Exception as e:
        st.error(f"Error: {e}")

