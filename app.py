import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
import gdown

# -------------------------------------------------
# Page config (must be first Streamlit command)
# -------------------------------------------------
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# -------------------------------------------------
# Model download + loading (RandomForest + OHE)
# -------------------------------------------------
MODEL_PATH = "rfrfinal.pkl"
DRIVE_FILE_ID = "1Upm6qfDH9e0AXnP_6gtGC_u7vg8AnVUQ"  # <-- replace with your real ID


def download_model_from_drive():
    """
    Download the model file from Google Drive if it does not exist locally.
    """
    if os.path.exists(MODEL_PATH):
        return

    st.warning("Model file not found locally. Downloading from Google Drive...")
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
    st.success("Model downloaded successfully.")


@st.cache_resource
def load_model():
    """
    Ensure the model is available locally and then load it from disk.
    """
    download_model_from_drive()
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


model = load_model()

# -------------------------------------------------
# App header
# -------------------------------------------------
st.title("🚗 Car Price Predictor")
st.write(
    "This app uses a trained Random Forest regression model "
    "to estimate the price of a vehicle."
)

st.markdown("---")

# -------------------------------------------------
# Input form
# -------------------------------------------------
st.subheader("Enter vehicle details")

with st.form("car_input_form"):
    col1, col2 = st.columns(2)

    with col1:
        brand = st.text_input("Brand", value="TOYOTA")
        model_name = st.text_input("Model", value="COROLLA")
        transmission = st.selectbox(
            "Transmission",
            options=["AUTO", "MANUAL"],
            index=0
        )

    with col2:
        fuel = st.selectbox(
            "Fuel Type",
            options=["PETROL", "DIESEL", "HYBRID", "OTHER"],
            index=0
        )
        capacity = st.number_input(
            "Engine Capacity (cc)",
            min_value=600,
            max_value=5000,
            value=1500,
            step=50
        )
        mileage = st.number_input(
            "Mileage (km)",
            min_value=0,
            max_value=300_000,
            value=50_000,
            step=1_000
        )

    submit = st.form_submit_button("Predict Price 💰")

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if submit:
    # Match your notebook preprocessing: everything uppercased
    brand_clean = brand.strip().upper()
    model_clean = model_name.strip().upper()
    transmission_clean = transmission.strip().upper()
    fuel_clean = fuel.strip().upper()

    # Build input row with EXACT column names used in training
    input_data = pd.DataFrame(
        {
            "Brand": [brand_clean],
            "Model": [model_clean],
            "Transmission": [transmission_clean],
            "Fuel": [fuel_clean],
            "Capacity": [capacity],
            "Mileage": [mileage],
        }
    )

    try:
        prediction = model.predict(input_data)
        price = float(prediction[0])

        st.markdown("### ✅ Predicted Price")
        st.success(f"Estimated price: **{price:,.0f}** (model units)")

        st.caption(
            "Note: This is a statistical estimate based on historical data. "
            "Real market prices may vary."
        )
    except Exception as e:
        st.error(f"Something went wrong during prediction: {e}")
        st.info(
            "Double-check that the column names in the DataFrame "
            "match those used when training the model."
        )

st.markdown("---")

# -------------------------------------------------
# Optional: batch prediction via CSV upload
# -------------------------------------------------
st.subheader("📄 Batch prediction (optional)")
st.write("Upload a CSV with the same columns as your training data (except `Price`).")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        if st.button("Predict for all rows"):
            preds = model.predict(df)
            df_result = df.copy()
            df_result["Predicted_Price"] = preds
            st.write("Results:")
            st.dataframe(df_result.head())
            st.download_button(
                "Download results as CSV",
                data=df_result.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"Error reading or predicting from CSV: {e}")
