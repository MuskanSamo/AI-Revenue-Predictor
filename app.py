import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# -----------------------------
# Load Model & Supporting Files
# -----------------------------
model = tf.keras.models.load_model("revenue_model.keras")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")
continuous_cols = joblib.load("continuous_cols.pkl")
binary_cols = joblib.load("binary_cols.pkl")

# -----------------------------
# Streamlit Page
# -----------------------------
st.set_page_config(
    page_title="AI Revenue Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI-Generated Media Revenue Predictor")

st.write(
    """
This application predicts the expected revenue of AI-generated media
using a Deep Learning Regression Model.
"""
)

st.markdown("---")

st.header("Enter Media Details")

# -----------------------------
# User Inputs
# -----------------------------

platform = st.selectbox(
    "Platform",
    ["TikTok", "Instagram", "YouTube"]
)

content_type = st.selectbox(
    "Content Type",
    ["Short", "Reel", "Video"]
)

ai_tool = st.selectbox(
    "AI Tool",
    [
        "Canva AI",
        "CapCut AI",
        "HeyGen",
        "InVideo AI",
        "Pika",
        "Runway",
        "Synthesia"
    ]
)

ai_type = st.selectbox(
    "AI Type",
    [
        "AI-assisted editing",
        "Avatar video",
        "Image-to-video",
        "Text-to-video"
    ]
)

duration = st.number_input(
    "Duration (seconds)",
    min_value=1.0,
    value=60.0
)

production_cost = st.number_input(
    "Production Cost (USD)",
    min_value=0.0,
    value=100.0
)

views = st.number_input(
    "Views",
    min_value=0.0,
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
    "Engagement Rate (%)",
    min_value=0.0,
    value=5.0
)

watch_time = st.number_input(
    "Average Watch Time (seconds)",
    min_value=0.0,
    value=45.0
)

retention_rate = st.number_input(
    "Retention Rate (%)",
    min_value=0.0,
    value=60.0
)
# -----------------------------
# Prediction
# -----------------------------

st.markdown("---")

if st.button("🚀 Predict Revenue"):

    # ---------- Engineered Features ----------
    engagement_score = likes + comments + shares

    views_per_sec = views / duration if duration > 0 else 0

    likes_ratio = likes / views if views > 0 else 0
    comments_ratio = comments / views if views > 0 else 0
    shares_ratio = shares / views if views > 0 else 0

    cost_per_view = production_cost / views if views > 0 else 0

    engagement_per_second = engagement_score / duration if duration > 0 else 0

    log_views = np.log1p(views)
    log_cost = np.log1p(production_cost)

    virality_score = (
        engagement_rate
        * retention_rate
        * shares_ratio
    )
    

    # ---------- Create Input ----------
    input_df = pd.DataFrame(0.0, index=[0], columns=columns)

    # Continuous Features
    input_df.loc[0, "duration_seconds"] = duration
    input_df.loc[0, "production_cost_usd"] = production_cost
    input_df.loc[0, "views"] = views
    input_df.loc[0, "likes"] = likes
    input_df.loc[0, "comments"] = comments
    input_df.loc[0, "shares"] = shares
    input_df.loc[0, "engagement_rate"] = engagement_rate
    input_df.loc[0, "watch_time_avg_sec"] = watch_time
    input_df.loc[0, "retention_rate"] = retention_rate

    input_df.loc[0, "engagement_score"] = engagement_score
    input_df.loc[0, "views_per_sec"] = views_per_sec
    input_df.loc[0, "likes_ratio"] = likes_ratio
    input_df.loc[0, "comments_ratio"] = comments_ratio
    input_df.loc[0, "shares_ratio"] = shares_ratio
    input_df.loc[0, "cost_per_view"] = cost_per_view
    input_df.loc[0, "engagement_per_second"] = engagement_per_second
    input_df.loc[0, "log_views"] = log_views
    input_df.loc[0, "log_cost"] = log_cost
    input_df.loc[0, "virality_score"] = virality_score

    # ---------- Binary Columns ----------

    if platform == "TikTok":
        input_df.loc[0, "platform_TikTok"] = 1

    elif platform == "YouTube":
        input_df.loc[0, "platform_YouTube"] = 1

    if content_type == "Short":
        input_df.loc[0, "content_type_Short"] = 1

    elif content_type == "Video":
        input_df.loc[0, "content_type_Video"] = 1

    if ai_type == "Avatar video":
        input_df.loc[0, "ai_type_Avatar video"] = 1

    elif ai_type == "Image-to-video":
        input_df.loc[0, "ai_type_Image-to-video"] = 1

    elif ai_type == "Text-to-video":
        input_df.loc[0, "ai_type_Text-to-video"] = 1

    tool_col = f"ai_tool_used_{ai_tool}"

    if tool_col in input_df.columns:
        input_df.loc[0, tool_col] = 1

    # ---------- Scale Continuous Features ----------
    input_df[continuous_cols] = scaler.transform(
        input_df[continuous_cols]
    )
    # ---------- Prediction ----------

    with st.spinner("🤖 AI is analyzing your media... Please wait..."):

        progress = st.progress(0)

        for i in range(100):
            progress.progress(i + 1)

        prediction = model.predict(input_df, verbose=0)

        progress.empty()

    revenue = prediction[0][0]

    st.success(f"💰 Predicted Revenue: ${revenue:,.2f}")

    st.balloons()

    st.caption(f"Raw Model Output: {prediction[0][0]:.4f}")
