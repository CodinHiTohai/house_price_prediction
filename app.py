import streamlit as st
import joblib
import pandas as pd
import os
import gdown

# Google Drive se model download (agar local me nahi hai)
MODEL_PATH = "model.pkl"
PIPELINE_PATH = "pipeline.pkl"

if not os.path.exists(MODEL_PATH):
    gdown.download(
        id="1pdluyd_SjNnD0iP-_U65vwzqbtGaY2Jt",
        output=MODEL_PATH,
        quiet=False
    )

# Pipeline bhi download karo (agar local me nahi hai)
# ⚠️ Apna pipeline.pkl ka Google Drive ID yahan daalo
PIPELINE_DRIVE_ID = "188_Pu181CgEZoCeYwmnVA2aa473LEtYg"
if not os.path.exists(PIPELINE_PATH) and PIPELINE_DRIVE_ID != "YOUR_PIPELINE_DRIVE_ID_HERE":
    gdown.download(
        id=PIPELINE_DRIVE_ID,
        output=PIPELINE_PATH,
        quiet=False
    )

# Model aur Pipeline load
model = joblib.load(MODEL_PATH)
pipeline = joblib.load(PIPELINE_PATH)

st.set_page_config(page_title="House Price Prediction", page_icon="🏠")

st.title("🏠 California House Price Prediction")
st.write("Enter the house details below.")

# Numeric Inputs
longitude = st.number_input("Longitude", value=-122.23)
latitude = st.number_input("Latitude", value=37.88)
housing_median_age = st.number_input("Housing Median Age", value=41)
total_rooms = st.number_input("Total Rooms", value=880)
total_bedrooms = st.number_input("Total Bedrooms", value=129)
population = st.number_input("Population", value=322)
households = st.number_input("Households", value=126)
median_income = st.number_input("Median Income", value=8.3252)

# Categorical Input
ocean = st.selectbox(
    "Ocean Proximity",
    [
        "<1H OCEAN",
        "INLAND",
        "ISLAND",
        "NEAR BAY",
        "NEAR OCEAN"
    ]
)

if st.button("Predict House Price"):

    # Raw column names use karo (pipeline khud transform kar legi)
    input_data = pd.DataFrame({
        "longitude": [longitude],
        "latitude": [latitude],
        "housing_median_age": [housing_median_age],
        "total_rooms": [total_rooms],
        "total_bedrooms": [total_bedrooms],
        "population": [population],
        "households": [households],
        "median_income": [median_income],
        "ocean_proximity": [ocean]
    })

    # ✅ Pehle pipeline se transform karo, phir predict karo
    transformed_input = pipeline.transform(input_data)
    prediction = model.predict(transformed_input)

    st.success(f"🏡 Predicted House Price: ${prediction[0]:,.2f}")