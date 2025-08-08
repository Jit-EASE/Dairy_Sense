# DairySense+ Prototype
# Streamlit + Gemini 2.5 + OpenAI API Joint Integration
# Author: Jit
# Version: 2.0

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
import base64
from datetime import datetime
from openai import OpenAI
from PIL import Image

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
st.title("🐄 DairySense+ | AI-Augmented Dairy Farming Assistant")
st.markdown(
    "Analyze dairy cow health and get AI-powered farm management recommendations "
    "using **Google Gemini 2.5** for vision and **OpenAI** for reasoning."
)

# File uploader
uploaded_img = st.file_uploader("Upload a cow or feed image", type=["jpg", "jpeg", "png"])

# Synthetic IoT Data Inputs
st.subheader("📡 Synthetic Barn Data (Demo)")
milk_yield = st.slider("Current Daily Milk Yield (litres)", 10, 40, 25)
cow_temp = st.slider("Average Cow Temperature (°C)", 37.0, 40.0, 38.5, 0.1)
feed_quality_score = st.slider("Feed Quality Score (1=Poor, 10=Excellent)", 1, 10, 8)
market_price = st.slider("Market Milk Price (€/litre)", 0.25, 1.50, 0.85, 0.01)

# --------------------
# Gemini 2.5 Image Analysis
# --------------------
def analyze_with_gemini(image_file):
    """
    Sends image to Gemini 2.5 API for analysis.
    """
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
# OpenAI Reasoning
# --------------------
def generate_openai_recommendations(gemini_analysis, milk_yield, cow_temp, feed_quality_score, market_price):
    """
    Uses OpenAI to combine Gemini's analysis + farm data into actionable recommendations.
    """
    prompt = f"""
    You are an AI dairy farm advisor.
    Gemini analysis:
    {gemini_analysis}

    Farm data:
    - Current milk yield: {milk_yield} L/day
    - Avg cow temperature: {cow_temp} °C
    - Feed quality score: {feed_quality_score}/10
    - Market milk price: €{market_price}/L

    Tasks:
    1. Summarize herd health status.
    2. Give 3 actionable recommendations for the next 48 hours.
    3. Forecast next week's milk yield using a simple OLS-like logic based on current trends.
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

    with st.spinner("Analyzing image with Gemini 2.5..."):
        gemini_result = analyze_with_gemini(uploaded_img)
    st.subheader("🔍 Gemini Image Analysis")
    st.code(gemini_result)

    if gemini_result != "No analysis available":
        with st.spinner("Generating recommendations with OpenAI..."):
            recommendations = generate_openai_recommendations(
                gemini_result, milk_yield, cow_temp, feed_quality_score, market_price
            )
        st.subheader("💡 AI Recommendations")
        st.markdown(recommendations)

# Footer
st.markdown("---")
st.caption("Prototype: Gemini 2.5 + OpenAI for Dairy Farming | Built by Jit")
