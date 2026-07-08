import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# -----------------------------
# Load Model & Files
# -----------------------------
model = tf.keras.models.load_model("revenue_model.keras")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Revenue Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Generated Media Revenue Predictor")

st.write("""
Predict the expected revenue of AI-generated media using a Deep Learning model.
""")

st.markdown("---")

st.header("Enter Media Details")

# -----------------------------
# User Inputs
# -----------------------------

platform = st.selectbox(
    "Platform",
    ["TikTok", "YouTube", "Instagram"]
)

content_type = st.selectbox(
    "Content Type",
    ["Short", "Reel", "Video"]
)

ai_tool = st.selectbox(
    "AI Tool",
    [
        "Adobe Firefly",
        "HeyGen",
        "Canva AI",
        "InVideo AI",
        "Pika",
        "Runway",
        "CapCut AI",
        "Synthesia"
    ]
)

ai_type = st.selectbox(
    "AI Type",
    [
        "Text-to-video",
        "Avatar video",
        "AI-assisted editing",
        "Image-to-video"
    ]
)

duration = st.number_input("Duration (seconds)", min_value=1.0, value=60.0)

production_cost = st.number_input(
    "Production Cost (USD)",
    min_value=1.0,
    value=100.0
)

views = st.number_input(
    "Views",
    min_value=1.0,
    value=1000.0
)

likes = st.number_input(
    "Likes",
    min_value=0.0,
    value=100.0
)

comments = st.number_input(
    "Comments",
    min_value=0.0,
    value=20.0
)

shares = st.number_input(
    "Shares",
    min_value=0.0,
    value=10.0
)

engagement_rate = st.number_input(
    "Engagement Rate",
    min_value=0.0,
    value=5.0
)

watch_time = st.number_input(
    "Average Watch Time (seconds)",
    min_value=0.0,
    value=45.0
)

retention_rate = st.number_input(
    "Retention Rate",
    min_value=0.0,
    value=60.0
)
