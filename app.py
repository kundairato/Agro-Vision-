	import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import datetime

st.set_page_config(page_title="Agro-Vision AI", layout="centered")

# -----------------------------
# Load AI Model
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=True
    )
    return model

model = load_model()

# -----------------------------
# Disease Recommendation System
# -----------------------------
def get_treatment(disease):
    treatments = {
        "leaf_blight": {
            "treatment": "Apply Mancozeb fungicide every 7 days.",
            "severity": "Medium"
        },
        "rust": {
            "treatment": "Use Propiconazole-based fungicide.",
            "severity": "High"
        },
        "healthy": {
            "treatment": "Crop is healthy. Maintain proper irrigation.",
            "severity": "Low"
        },
        "unknown": {
            "treatment": "Consult agricultural extension officer.",
            "severity": "Unknown"
        }
    }
    return treatments.get(disease, treatments["unknown"])

# -----------------------------
# Image Processing
# -----------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

# -----------------------------
# UI
# -----------------------------
st.title("🌿 Agro-Vision 2.0")
st.subheader("AI-Powered Crop Disease Detection System")

menu = st.sidebar.selectbox(
    "Navigation",
    ["Home", "Scan Crop", "Dashboard", "About"]
)

# -----------------------------
# HOME
# -----------------------------
if menu == "Home":
    st.write("### Welcome to Agro-Vision")
    st.write("Detect crop diseases instantly using AI.")
    st.success("Works Online | Smart Detection | Treatment Advice")

# -----------------------------
# SCAN PAGE
# -----------------------------
elif menu == "Scan Crop":
    uploaded_file = st.file_uploader(
        "Upload a crop leaf image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Analyze Crop"):
            with st.spinner("AI analyzing..."):
                processed = preprocess_image(image)
                predictions = model.predict(processed)

                decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)
                predicted_label = decoded[0][0][1]
                confidence = decoded[0][0][2] * 100

                if "leaf" in predicted_label:
                    disease = "leaf_blight"
                elif "rust" in predicted_label:
                    disease = "rust"
                else:
                    disease = "healthy"

                result = get_treatment(disease)

            st.success("Analysis Complete")

            st.write("### Detection Result")
            st.write(f"**Disease:** {disease}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            st.write(f"**Severity:** {result['severity']}")

            st.write("### Treatment Recommendation")
            st.info(result["treatment"])

            st.write("### Scan Timestamp")
            st.write(datetime.datetime.now())

# -----------------------------
# DASHBOARD
# -----------------------------
elif menu == "Dashboard":
    st.write("### Crop Health Dashboard")
    st.metric("Total Scans (Demo)", 12)
    st.metric("Healthy Crops", "75%")
    st.metric("High Severity Cases", "2")

    st.write("Future Version: Real-time analytics & outbreak map")

# -----------------------------
# ABOUT
# -----------------------------
elif menu == "About":
    st.write("""
    Agro-Vision is an AI-based crop disease detection system 
    built using Computer Vision and Deep Learning.
    
    Version: 2.0
    Developer: Kundai Rato
    """)

st.markdown("---")
st.caption("Agro-Vision © 2026")