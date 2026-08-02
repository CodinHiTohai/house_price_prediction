import streamlit as st
import joblib
import pandas as pd
import os
import gdown

# Google Drive se model download (agar local me nahi hai)
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    gdown.download(
        id="1pdluyd_SjNnD0iP-_U65vwzqbtGaY2Jt",
        output=MODEL_PATH,
        quiet=False
    )

# Model load
model = joblib.load(MODEL_PATH)

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

    input_data = pd.DataFrame({
        "num__longitude": [longitude],
        "num__latitude": [latitude],
        "num__housing_median_age": [housing_median_age],
        "num__total_rooms": [total_rooms],
        "num__total_bedrooms": [total_bedrooms],
        "num__population": [population],
        "num__households": [households],
        "num__median_income": [median_income],
        "cat__ocean_proximity_<1H OCEAN": [1 if ocean == "<1H OCEAN" else 0],
        "cat__ocean_proximity_INLAND": [1 if ocean == "INLAND" else 0],
        "cat__ocean_proximity_ISLAND": [1 if ocean == "ISLAND" else 0],
        "cat__ocean_proximity_NEAR BAY": [1 if ocean == "NEAR BAY" else 0],
        "cat__ocean_proximity_NEAR OCEAN": [1 if ocean == "NEAR OCEAN" else 0]
    })

    prediction = model.predict(input_data)

    st.success(f"🏡 Predicted House Price: ${prediction[0]:,.2f}")