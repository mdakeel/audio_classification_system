import os
import streamlit as st
import torchaudio
import matplotlib.pyplot as plt
import joblib
from PIL import Image

from src.pipeline.prediction_pipeline import SinglePrediction
from src.pipeline.training_pipeline import TrainingPipeline
from src.constants import STATIC_DIR, UPLOAD_SUB_DIR

# Set page config
st.set_page_config(page_title="Cat Dog Audio Classifier", layout="centered")

# Title
st.title("🐱🐶 Cat Dog Audio Classifier")

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Home", "Predict", "Train"])

# ---------------- Home Page ----------------
if page == "Home":
    st.header("Welcome to Cat Dog Audio Classifier")
    st.write(
        "Upload an audio file to classify whether it contains a cat or dog sound, "
        "or train the model with a single click!"
    )

    # Build absolute path for hero image
    hero_path = os.path.join(os.getcwd(), "static", "images", "hero-bg.jpg")
    if os.path.exists(hero_path):
        st.image(hero_path, width=600)
    else:
        st.warning("Hero image not found at static/images/hero-bg.jpg")

# ---------------- Prediction Page ----------------
elif page == "Predict":
    st.header("Upload Audio for Prediction")

    uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])
    if uploaded_file is not None:
        # Save uploaded file
        upload_dir = os.path.join(STATIC_DIR, UPLOAD_SUB_DIR)
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, "upload_file.wav")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        st.audio(file_path, format="audio/wav")

        # Run prediction
        pred = SinglePrediction()
        result = pred.predict()

        # Show spectrogram
        st.subheader("Spectrogram")
        spec_path = os.path.join(upload_dir, "image.jpg")
        if os.path.exists(spec_path):
            st.image(spec_path, width=400)
        else:
            st.warning("Spectrogram image not found")

        # Show result
        st.subheader("Prediction Result")
        st.success(result.upper())

# ---------------- Training Page ----------------
elif page == "Train":
    st.header("Model Training")
    st.write(
        "Initiate the training process for the Cat Dog Audio Classifier. "
        "This will train a machine learning model to distinguish between cat and dog sounds."
    )
    st.warning("Once training starts, it cannot be stopped until completion.")

    if st.button("Start Training"):
        with st.spinner("Training in progress... this may take some time"):
            pipeline = TrainingPipeline()
            pipeline.run_pipeline()
        st.success("Training Completed!")
