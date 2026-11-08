import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Loading the saved model and feature list
pipeline = joblib.load('models/kepler_exoplanet_pipeline.pkl')
feature_names = joblib.load('models/feature_names.pkl')

st.set_page_config(page_title="NASA Kepler Exoplanet Classifier", layout="wide")
st.title("🪐 NASA Kepler Exoplanet Search & Classification")
st.write("A system that classifies objects as a real exoplanet (`CONFIRMED`) or a false alarm (`FALSE POSITIVE`) based on transit photometry measurements.")

st.sidebar.header("Input Planetary & Stellar Parameters")

# Receiving key features from the user
koi_period = st.sidebar.number_input("Orbital Period (days)", min_value=0.1, value=10.5)
koi_duration = st.sidebar.number_input("Transit Duration (hours)", min_value=0.1, value=3.2)
koi_depth = st.sidebar.number_input("Transit Depth (ppm)", min_value=1.0, value=500.0)
koi_prad = st.sidebar.number_input("Planetary Radius (Earth radii)", min_value=0.1, value=1.5)
koi_teq = st.sidebar.number_input("Equilibrium Temperature (K)", min_value=50.0, value=280.0)
koi_srad = st.sidebar.number_input("Stellar Radius (Solar radii)", min_value=0.1, value=1.0)
koi_model_snr = st.sidebar.number_input("Transit Signal-to-Noise Ratio (SNR)", min_value=1.0, value=25.0)

if st.sidebar.button("Predict Disposition"):
    # Creating a DataFrame with default NaNs for all features (the imputer will fill the rest with the median)
    input_data = pd.DataFrame(columns=feature_names)
    input_data.loc[0] = np.nan

    # Assigning the values provided by the user
    input_data['koi_period'] = koi_period
    input_data['koi_duration'] = koi_duration
    input_data['koi_depth'] = koi_depth
    input_data['koi_prad'] = koi_prad
    input_data['koi_teq'] = koi_teq
    input_data['koi_srad'] = koi_srad
    input_data['koi_model_snr'] = koi_model_snr

    # Calculating engineered features
    if 'radius_ratio' in feature_names:
        input_data['radius_ratio'] = koi_prad / (koi_srad * 109.1)
    if 'transit_duty_cycle' in feature_names:
        input_data['transit_duty_cycle'] = (koi_duration / 24.0) / koi_period
    if 'depth_to_snr_ratio' in feature_names:
        input_data['depth_to_snr_ratio'] = koi_depth / (koi_model_snr + 1e-5)

    # Making predictions
    pred = pipeline.predict(input_data)[0]
    prob = pipeline.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")
    if pred == 1:
        st.success(f"✅ **CONFIRMED EXOPLANET** (Confidence: {prob*100:.2f}%)")
        if 0.8 <= koi_prad <= 2.0 and 180 <= koi_teq <= 310:
            st.info("🌍 **Habitable Zone Alert:** This planet is close to Earth's size and lies within a potentially habitable temperature range!")
    else:
        st.error(f"❌ **FALSE POSITIVE** (Exoplanet Probability: {prob*100:.2f}%)")
