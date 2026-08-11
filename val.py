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
st.markdown("Validate the labels of your uploaded dataset using our model.")


welcome_placeholder = st.empty()
with welcome_placeholder.container():
    st_lottie(welcome_animation, height=600, key="welcome")


import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.metrics import confusion_matrix
import numpy as np
import re

st.subheader("📤 Validate Labels of Uploaded Dataset")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="validate_csv")

def auto_detect_text_label(df):

    text_column = None
    label_column = None

    for col in df.columns:
       
        if df[col].dtype == 'object' and df[col].apply(lambda x: isinstance(x, str) and len(x.split(',')) > 1).any():
            if not text_column:
                text_column = col
        
       
        if df[col].dtype == 'object' and df[col].apply(lambda x: isinstance(x, str) and re.match(r'^(positive|negative|neutral|1|-1|0)$', x.strip().lower()) is not None).any():
            if not label_column:
                label_column = col

    return text_column, label_column

if uploaded_file:
    try:
        df_upload = pd.read_csv(uploaded_file)

        text_column, label_column = auto_detect_text_label(df_upload)

        if not text_column or not label_column:
            st.error("Could not automatically detect 'text' and 'label' columns.")
        else:
            st.success(f"File uploaded and validated successfully ✅")

            st.markdown("### 📊 Uploaded Label Distribution")
            st.bar_chart(df_upload[label_column].value_counts())

            df_upload['preprocessed_text'] = df_upload[text_column].apply(preprocess_text)
            df_upload['model_prediction'] = predict(df_upload['preprocessed_text'].tolist())

            df_upload['agreement'] = df_upload[label_column] == df_upload['model_prediction']
            total = len(df_upload)
            matches = df_upload['agreement'].sum()
            accuracy = matches / total if total > 0 else 0
            st.metric("✅ Label Agreement with Model", f"{accuracy * 100:.2f}%")

            st.markdown("### 📈 Agreement Summary")
            st.plotly_chart(px.pie(
                df_upload,
                names=['agreement' if val else 'mismatch' for val in df_upload['agreement']],
                title='Label vs Model Agreement',
                hole=0.5,
                color_discrete_sequence=["#00C49F", "#FF8042"]
            ), use_container_width=True)

         
            st.markdown("### 🧪 Per-Class Accuracy")
            class_accuracy = df_upload.groupby(label_column).apply(
                lambda x: (x[label_column] == x['model_prediction']).mean()
            ).sort_values(ascending=False)
            st.bar_chart(class_accuracy)

       
            mismatches = df_upload[~df_upload['agreement']]
            st.markdown("### 🔍 Mismatched Labels")
            if mismatches.empty:
                st.success("🎯 All labels match the model's predictions!")
            else:
                selected_label = st.selectbox("Filter mismatches by actual label", ['All'] + sorted(mismatches[label_column].unique().tolist()))
                if selected_label != 'All':
                    mismatches = mismatches[mismatches[label_column] == selected_label]

                st.dataframe(mismatches[[text_column, label_column, 'model_prediction']], use_container_width=True)

    
            st.markdown("### 🔄 Confusion Matrix")
            cm = confusion_matrix(df_upload[label_column], df_upload['model_prediction'], labels=np.unique(df_upload[label_column]))
            cm_df = pd.DataFrame(cm, index=np.unique(df_upload[label_column]), columns=np.unique(df_upload[label_column]))
            st.dataframe(cm_df.style.background_gradient(cmap='Blues'), use_container_width=True)

         
            st.markdown("### 📥 Download Results")
            results_csv = df_upload.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download CSV with Validation Info",
                data=results_csv,
                file_name="validated_dataset.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"❌ Failed to process file: {e}")

