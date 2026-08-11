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
    with open(file_path, "r", encoding="utf-8") as f: 
        return json.load(f)


loading_animation = load_lottie_animation("loading.json")
welcome_animation = load_lottie_animation("welcome.json")

st.set_page_config(page_title="SentiScrape Dashboard", layout="wide")
st.title("🧠 Reddit Brand SentiScrape Dashboard")
st.markdown("Analyze Reddit posts related to a brand using Transformer-based sentiment classification.")
st_lottie(welcome_animation, height=500, key="welcome")

st.subheader("CHOOSE YOUR WORK:")
col1, col2 = st.columns(2)

with col1:
    if st.button("  SCRAP WITH BRAND NAME  ", key="scrap_button"):
        st.success("Navigating to Main Page...")
        os.system("streamlit run reddit_scrap.py")
    st.markdown("<small style='color: grey;'>Find market sentiments of your Brand(On Reddit).</small>", unsafe_allow_html=True)

with col2:
    if st.button("  VALIDATE DATASET  ", key="val_button"):
        st.success("Navigating to Validation Page...")
        os.system("streamlit run val.py")
    st.markdown("<small style='color: grey;'>Find accuracy of your Data</small>", unsafe_allow_html=True)