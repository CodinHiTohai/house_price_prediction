import streamlit as st
import pandas as pd
import numpy as np
import gdown
import os

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# ─── Housing CSV download (agar cloud pe nahi hai) ───────────────────────────
CSV_PATH = "housing.csv"
CSV_DRIVE_ID = "1pdluyd_SjNnD0iP-_U65vwzqbtGaY2Jt"  # apna housing.csv Drive ID

if not os.path.exists(CSV_PATH):
    gdown.download(id=CSV_DRIVE_ID, output=CSV_PATH, quiet=False)

# ─── Model + Pipeline train karo (sirf ek baar, cache mein rahega) ───────────
@st.cache_resource
def load_model():
    housing = pd.read_csv(CSV_PATH)

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, _ in split.split(housing, housing["income_cat"]):
        housing = housing.loc[train_index].drop("income_cat", axis=1)

    housing_labels = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value", axis=1)

    num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])

    housing_prepared = full_pipeline.fit_transform(housing_features)

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)

    return model, full_pipeline

# ─── App UI ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="House Price Prediction", page_icon="🏠")
st.title("🏠 California House Price Prediction")
st.write("Enter the house details below.")

with st.spinner("⏳ Model load ho raha hai... pehli baar thoda time lagega."):
    model, pipeline = load_model()

# Numeric Inputs
longitude          = st.number_input("Longitude",            value=-122.23)
latitude           = st.number_input("Latitude",             value=37.88)
housing_median_age = st.number_input("Housing Median Age",   value=41)
total_rooms        = st.number_input("Total Rooms",          value=880)
total_bedrooms     = st.number_input("Total Bedrooms",       value=129)
population         = st.number_input("Population",           value=322)
households         = st.number_input("Households",           value=126)
median_income      = st.number_input("Median Income",        value=8.3252)

# Categorical Input
ocean = st.selectbox(
    "Ocean Proximity",
    ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
)

if st.button("Predict House Price"):
    input_data = pd.DataFrame({
        "longitude":          [longitude],
        "latitude":           [latitude],
        "housing_median_age": [housing_median_age],
        "total_rooms":        [total_rooms],
        "total_bedrooms":     [total_bedrooms],
        "population":         [population],
        "households":         [households],
        "median_income":      [median_income],
        "ocean_proximity":    [ocean]
    })

    transformed_input = pipeline.transform(input_data)
    prediction = model.predict(transformed_input)

    st.success(f"🏡 Predicted House Price: ${prediction[0]:,.2f}")