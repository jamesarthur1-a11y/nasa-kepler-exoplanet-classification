# 🪐 NASA Kepler Exoplanet Search & Classification

An end-to-end Data Science and Machine Learning project that classifies Kepler Objects of Interest (KOIs) into **Confirmed Exoplanets** vs. **Astrophysical False Positives** using transit photometry measurements.

## 📌 Project Overview
- **Dataset:** NASA Kepler Exoplanet Search Results (~9,564 observations, 50 columns)
- **Objective:** Build a leak-free binary classification pipeline based on physical transit and stellar parameters, and use the trained model to evaluate unconfirmed planetary `CANDIDATE` signals.
- **Deployment:** Interactive web application built with **Streamlit**.

## 🛠️ Methodology & Pipeline
1. **Data Cleaning & Leakage Prevention:** Removed automated pipeline vetting scores (`koi_score`, `koi_pdisposition`) and uncertainty columns (`_err1`, `_err2`) to prevent target leakage.
2. **Exploratory Data Analysis & Hypothesis Testing:** Evaluated right-skewed transit distributions and performed the **Mann-Whitney U Test** ($p < 0.05$) confirming statistically significant differences in transit depth between confirmed exoplanets and false positives.
3. **Astrophysical Feature Engineering:**
   - `radius_ratio`: Planet-to-Star radius ratio
   - `transit_duty_cycle`: Ratio of transit duration to orbital period
   - `depth_to_snr_ratio`: Normalized transit depth
4. **Preprocessing:** Median imputation paired with `RobustScaler` to handle extreme astrophysical outliers.
5. **Model Training & Tuning:** Compared Logistic Regression, Random Forest, and Gradient Boosting; optimized via `RandomizedSearchCV` with 5-Fold Stratified Cross-Validation.

## 🚀 How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py