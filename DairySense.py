# DairySense+ Prototype
# Streamlit + Gemini 2.5 + OpenAI + OLS Forecasting (Plotly Interactive)
# Author: Jit
# Version: 2.6

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
import base64
from datetime import datetime, timedelta
from openai import OpenAI
from PIL import Image
import statsmodels.api as sm
import plotly.graph_objects as go

# --------------------
# API Keys (Streamlit Secrets)
# --------------------
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Init OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# --------------------
# Streamlit UI
# --------------------
st.set_page_config(page_title="DairySense+ AI", layout="wide")
st.title("DairySense+ | Smart Dairy Farming Assistant for Ireland")
st.markdown(
    "Analyze dairy cow health, forecast milk yield, and get farm recommendations "
    "(Powered by Econometrics and AI)"
)

# File uploader
uploaded_img = st.file_uploader("Upload a cow or feed image", type=["jpg", "jpeg", "png"])

# Synthetic IoT Data Inputs
st.subheader("IoT Sensor based Barn Data")
milk_yield = st.slider("Current Daily Milk Yield (litres)", 10, 40, 25)
cow_temp = st.slider("Average Cow Temperature (°C)", 37.0, 40.0, 38.5, 0.1)
feed_quality_score = st.slider("Feed Quality Score (1=Poor, 10=Excellent)", 1, 10, 8)
market_price = st.slider("Market Milk Price (€/litre)", 0.25, 1.50, 0.85, 0.01)

# --------------------
# Gemini 2.5 Image Analysis
# --------------------
def analyze_with_gemini(image_file):
    url = "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}

    image_bytes = image_file.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            "Analyze the image of the cow or feed. Detect any visible health issues, "
                            "signs of disease, or feed quality problems. Provide concise structured JSON "
                            "with 'condition', 'confidence', and 'notes'."
                        )
                    },
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_base64
                        }
                    }
                ]
            }
        ]
    }

    response = requests.post(url, headers=headers, params=params, data=json.dumps(payload))
    data = response.json()

    if "candidates" not in data:
        st.error("⚠️ Gemini API returned no candidates. Showing raw response below for debugging.")
        st.code(json.dumps(data, indent=2))
        return "No analysis available"

    return data["candidates"][0]["content"]["parts"][0]["text"]

# --------------------
# OLS Forecasting with Plotly
# --------------------
def ols_forecast_plotly(current_yield):
    """
    Generate synthetic past yield data and forecast 7 days ahead using OLS regression.
    Returns forecast DataFrame and displays Plotly chart.
    """
    np.random.seed(42)
    days = np.arange(1, 31)
    yields = current_yield + np.sin(days/5) * 2 + np.random.normal(0, 0.5, len(days))

    # Create DataFrame
    df = pd.DataFrame({"day": days, "yield": yields})

    # OLS regression
    X = sm.add_constant(df["day"])
    model = sm.OLS(df["yield"], X).fit()

    # Forecast next 7 days
    future_days = np.arange(31, 38)
    X_future = sm.add_constant(future_days)
    forecast = model.predict(X_future)

    forecast_df = pd.DataFrame({"day": future_days, "forecast_yield": forecast})

    # Plotly chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["day"], y=df["yield"], mode='lines+markers',
        name="Historical Yield", line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df["day"], y=forecast_df["forecast_yield"], mode='lines+markers',
        name="Forecast", line=dict(color='orange', dash='dash')
    ))

    fig.update_layout(
        title="Milk Yield Forecast (OLS)",
        xaxis_title="Day",
        yaxis_title="Milk Yield (litres)",
        template="plotly_white",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    return forecast_df

# --------------------
# OpenAI Reasoning
# --------------------
def generate_openai_recommendations(gemini_analysis, forecast_df, milk_yield, cow_temp, feed_quality_score, market_price):
    forecast_text = forecast_df.to_string(index=False)
    prompt = f"""
    You are an AI dairy farm advisor.
    Gemini analysis:
    {gemini_analysis}

    Milk yield forecast for next 7 days (litres/day):
    {forecast_text}

    Current farm data:
    - Current milk yield: {milk_yield} L/day
    - Avg cow temperature: {cow_temp} °C
    - Feed quality score: {feed_quality_score}/10
    - Market milk price: €{market_price}/L

    Tasks:
    1. Summarize herd health status.
    2. Give 3 actionable recommendations for the next 48 hours.
    3. Interpret the OLS forecast and comment on trends.
    4. Suggest policy/sustainability considerations for Irish dairy farmers.
    """
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return completion.choices[0].message.content

# --------------------
# Main App Logic
# --------------------
if uploaded_img:
    st.image(uploaded_img, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        gemini_result = analyze_with_gemini(uploaded_img)
    st.subheader("Image Analysis")
    st.code(gemini_result)

    if gemini_result != "No analysis available":
        st.subheader("OLS Regression - Milk Yield Forecast")
        forecast_df = ols_forecast_plotly(milk_yield)

        with st.spinner("Generating recommendations..."):
            recommendations = generate_openai_recommendations(
                gemini_result, forecast_df, milk_yield, cow_temp, feed_quality_score, market_price
            )
        st.subheader(" Recommendations")
        st.markdown(recommendations)

# Footer
st.markdown("---")
st.caption("Prototype: Multi-Modal AI (Vision + Reasoning) & Econometrics | Built by Jit")
