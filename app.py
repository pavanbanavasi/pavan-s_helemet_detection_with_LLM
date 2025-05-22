import streamlit as st
import requests
import base64
import os
import tempfile
import cv2
from PIL import Image
import numpy as np

# === Configuration ===
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def analyze_helmet_with_groq(image_path):
    with open(image_path, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode()

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
You are a safety AI. Look at the uploaded image and answer the following:

- Is the person wearing a helmet? (Yes/No)

If unclear, respond with: "Uncertain".
"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.2,
        "max_tokens": 200
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content'].strip()
    else:
        return f"API Error: {response.status_code} - {response.text}"

# === Streamlit UI ===
st.set_page_config(page_title="Helmet Detection with Groq", layout="centered")
st.title("🪖 Helmet Detection Using Groq Vision")

uploaded_file = st.file_uploader("Upload an image (person with/without helmet)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.image(Image.open(tmp_path), caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        helmet_result = analyze_helmet_with_groq(tmp_path)

    # Load image using OpenCV and draw text
    image_cv = cv2.imread(tmp_path)
    label = f"Helmet: {helmet_result}"
    #cv2.putText(image_cv, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Convert BGR to RGB for Streamlit
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    st.markdown("### 🖼️ Result with Label:")
    st.image(image_rgb, caption=label, use_container_width=True)
