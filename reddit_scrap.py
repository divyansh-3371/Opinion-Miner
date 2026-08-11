import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random
import os
from datetime import datetime
from scraper import scrape_search_results, save_all_preprocessed_text, preprocess_text
from inference import predict
import google.generativeai as genai
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import json

def load_lottie_animation(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:  # Explicitly specify encoding as UTF-8
        return json.load(f)


loading_animation = load_lottie_animation("loading.json")
welcome_animation = load_lottie_animation("welcome.json")

st.set_page_config(page_title="SentiScrape Dashboard", layout="wide")
st.title("🧠 Reddit Brand SentiScrape Dashboard")
st.markdown("Analyze Reddit posts related to a brand using Transformer-based sentiment classification.")
welcome_placeholder = st.empty()
with welcome_placeholder.container():
    st_lottie(welcome_animation, height=500, key="welcome")


st.sidebar.header("🔧 Scraping & Filter Settings")
brand = st.sidebar.text_input("Enter Brand Name", placeholder="e.g., Nike")
limit = st.sidebar.slider("Number of Posts per Type (TOP & HOT)", min_value=30, max_value=5000, step=10, value=100)

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime(2024, 1, 1))
with col2:
    end_date = st.date_input("End Date", value=datetime.now().date())

start_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
end_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())

min_length = st.sidebar.slider("Minimum Word Count", min_value=0, max_value=100, value=5)
sentiment_filter = st.sidebar.selectbox("Filter by Sentiment", ["All", "Positive", "Neutral", "Negative"])


categories = {
    'Product/Service Quality': ['quality', 'service', 'product'],
    'Customer Experience & Support': ['support', 'customer service'],
    'Brand Authenticity & Transparency': ['authenticity', 'transparency'],
    'Marketing & Social Presence': ['marketing', 'social media'],
    'Workplace Culture & Employee Satisfaction': ['workplace', 'culture', 'employees'],
    'Innovation & Adaptability': ['innovation', 'adaptability'],
    'Pricing & Value Perception': ['price', 'value'],
    'Scandals & Controversies': ['scandal', 'controversy'],
    'Community Engagement & Brand Loyalty': ['community', 'loyalty']
}
category_filter = st.sidebar.selectbox("Filter by Category", ["All"] + list(categories.keys()))


show_gemini_summary = st.sidebar.checkbox("Show Gemini Summary", value=True)

genai.configure(api_key="AIzaSyA7ZZx4zhDkw90jiMS-Xy9oRnBfBYC-OfE")  

def get_gemini_summary(texts):
    # Truncate to fit token limits
    MAX_TOTAL_CHARS = 12000
    truncated_texts, char_count = [], 0
    for t in texts:
        if char_count + len(t) > MAX_TOTAL_CHARS:
            break
        truncated_texts.append(t)
        char_count += len(t)

    summary_prompt = (
    "You are a sentiment analysis expert. Analyze the following Reddit posts about a brand and provide:\n"
    "1. A concise summary of the main topics and discussions.\n"
    "2. The overall sentiment trend (positive, negative, or neutral) and how strong it is.\n"
    "3. What this sentiment trend means for the brand and how it might affect brand reputation, loyalty, or trust.\n"
    "4. Why these sentiment insights are important for understanding customer perception.\n\n"
    "Here are the posts:\n\n"
    + "\n\n".join(truncated_texts)
    + "\n\nPlease return your summary and explanation in a clear, structured format."
    )
    try:
        model = genai.GenerativeModel("gemini-2.0-flash") 
        response = model.generate_content(summary_prompt)
        return {"summary": response.text.strip()}
    except Exception as e:
        return {"summary": f"Failed to fetch summary: {e}"}


def get_gemini_advice(texts):
    
    MAX_TOTAL_CHARS = 12000
    truncated_texts, char_count = [], 0
    for t in texts:
        if char_count + len(t) > MAX_TOTAL_CHARS:
            break
        truncated_texts.append(t)
        char_count += len(t)

    suggestion_prompt = (
    "You are a brand strategy expert. Based on the following Reddit posts, provide:\n"
    "1. Practical recommendations the brand should consider to improve or maintain its public image.\n"
    "2. Actions to address recurring complaints or concerns if the sentiment is negative.\n"
    "3. Opportunities the brand can leverage if the sentiment is positive.\n"
    "4. Advice on how to better engage with the community and strengthen brand trust.\n\n"
    "Here are the posts:\n\n"
    + "\n\n".join(truncated_texts)
    + "\n\nPlease format your advice in a list of prioritized, actionable suggestions."
    )

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")  
        response = model.generate_content(suggestion_prompt)
        return {"summary": response.text.strip()}
    except Exception as e:
        return {"summary": f"Failed to fetch summary: {e}"}


def safe_value_counts(df, column):
    return df[column].value_counts() if column in df.columns else pd.Series()

def generate_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(text))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=18)
    ax.axis("off")
    st.pyplot(fig)


if st.sidebar.button("Start Scraping & Analyzing"):
    if not brand:
        st.warning("Please enter a brand name.")
    else:
        welcome_placeholder.empty()
        loading_placeholder = st.empty()
        with loading_placeholder.container():
            anim_col1, anim_col2, anim_col3 = st.columns([1, 2, 1])
            with anim_col2:
                st_lottie(loading_animation, height=200, key="loading")

        
        top_results = scrape_search_results(brand, "top", limit, start_timestamp, end_timestamp)
        hot_results = scrape_search_results(brand, "hot", limit, start_timestamp, end_timestamp)
        all_results = top_results + hot_results

        loading_placeholder.empty()

        if not all_results:
            st.error("No posts found for the given criteria.")
        else:
            save_all_preprocessed_text(all_results, brand)
            texts = [preprocess_text(f"{post['title']} {post['body']}") for post in all_results]
            filtered_texts = [t for t in texts if len(t.split()) > min_length]
            sentiments = predict(filtered_texts)

            df = pd.DataFrame(all_results[:len(sentiments)])
            df['preprocessed_text'] = filtered_texts
            df['Length'] = df['preprocessed_text'].apply(lambda x: len(x.split()))
            df['Sentiment'] = sentiments

            if sentiment_filter != "All":
                df = df[df['Sentiment'] == sentiment_filter]
            df = df[df['Length'] >= min_length]

            if category_filter != "All":
                keywords = categories[category_filter]
                df = df[df['preprocessed_text'].str.contains('|'.join(keywords), case=False, na=False)]

           
            st.subheader("📋 Top 10 Sentiment Predictions")
            search_term = st.text_input("🔍 Search in Titles")
            display_df = df[['title', 'Sentiment']]
            if search_term:
                display_df = display_df[display_df['title'].str.contains(search_term, case=False)]
            st.dataframe(display_df.head(10), use_container_width=True)

         
            csv_path = f"reddit_posts/{brand}_predictions.csv"
            os.makedirs("reddit_posts", exist_ok=True)
            df.to_csv(csv_path, index=False)
            st.download_button("📁 Download CSV", data=df.to_csv(index=False), file_name=f"{brand}_predictions.csv", mime="text/csv")

           
            st.subheader("📊 Sentiment Overview")
            sentiment_counts = safe_value_counts(df, 'Sentiment').reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]
            pie_chart = px.pie(sentiment_counts, names="Sentiment", values="Count", title="Sentiment Distribution", color_discrete_sequence=px.colors.qualitative.Set2)
            bar_chart = px.bar(sentiment_counts, x="Sentiment", y="Count", title="Sentiment Distribution (Bar)", color="Sentiment", color_discrete_sequence=px.colors.qualitative.Set2)
            col1, col2 = st.columns(2)
            col1.plotly_chart(pie_chart, use_container_width=True)
            col2.plotly_chart(bar_chart, use_container_width=True)

           
            st.subheader("📊 Category-wise Sentiment Breakdown")
            category_sentiment_df = df.copy()
            category_sentiment_df['Category'] = category_filter if category_filter != "All" else category_sentiment_df['preprocessed_text'].apply(
                lambda text: next((cat for cat, keys in categories.items() if any(k in text.lower() for k in keys)), 'Uncategorized')
            )
            cat_sent_grouped = category_sentiment_df.groupby(['Category', 'Sentiment']).size().reset_index(name='Count')
            cat_sent_chart = px.bar(cat_sent_grouped, x="Category", y="Count", color="Sentiment", barmode="group", title="Category-wise Sentiment Breakdown")
            st.plotly_chart(cat_sent_chart, use_container_width=True)

           
            st.subheader("☁️ Word Clouds by Sentiment")
            for sentiment in ["Positive", "Neutral", "Negative"]:
                sentiment_texts = df[df['Sentiment'] == sentiment]['preprocessed_text']
                if not sentiment_texts.empty:
                    generate_wordcloud(sentiment_texts, f"{sentiment} Sentiment Word Cloud")

          
            st.subheader("📝 Summary")
            st.markdown(f"""
            - Most common sentiment: **{df['Sentiment'].mode()[0] if not df.empty else 'N/A'}**
            - Sentiment filtering: **{sentiment_filter}**  
            - Category filter: **{category_filter}**  
            - Minimum word count filter: **{min_length} words**
            """)

            if show_gemini_summary:
                sentiment_summary = get_gemini_summary(filtered_texts)
                if sentiment_summary:
                    st.subheader("📑 Sentiment Summary")
                    st.write(sentiment_summary['summary'])
                SENTIMENT_ADVICE = get_gemini_advice(filtered_texts)
                if SENTIMENT_ADVICE:
                    st.subheader("💡 Brand Strategy Advice")
                    st.write(SENTIMENT_ADVICE['summary'])