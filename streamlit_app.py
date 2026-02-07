import streamlit as st
import requests

# --------------------------------------------------

# CONFIG

# --------------------------------------------------

API_URL = "https://fake-news-detection-api-1.onrender.com/api/predict/"

st.set_page_config(
page_title="Fake News Detector",
page_icon="📰",
layout="centered"
)

# --------------------------------------------------

# UI HEADER

# --------------------------------------------------

st.title("📰 Fake News Detection System")
st.markdown(
"Check whether a news article is **Real or Fake** using ML."
)

# --------------------------------------------------

# INPUT OPTIONS

# --------------------------------------------------

option = st.radio(
"Choose Input Type",
("Text", "URL")
)

text_input = ""
url_input = ""

if option == "Text":
    text_input = st.text_area(
    "Enter News Text",
    height=200,
    placeholder="Paste news content here..."
    )

else:
    url_input = st.text_input(
    "Enter News Article URL",
    placeholder="[https://example.com/news](https://example.com/news)"
    )

# --------------------------------------------------

# PREDICT BUTTON

# --------------------------------------------------

if st.button("Detect News"):

    if not text_input and not url_input:
        st.warning("Please enter text or URL")
        st.stop()

    payload = {
        "text": text_input,
        "url": url_input
    }

    try:
        with st.spinner("Analyzing news..."):

            response = requests.post(API_URL, json=payload)

            if response.status_code == 200:
                data = response.json()

                prediction = data["prediction"]
                confidence = data["confidence"]

                if prediction == "REAL":
                    st.success(
                        f"🟢 REAL News\nConfidence: {confidence:.2%}"
                    )
                else:
                    st.error(
                        f"🔴 FAKE News\nConfidence: {confidence:.2%}"
                    )

            else:
                st.error("API Error")

    except Exception as e:
        st.error("Server not reachable")

