import streamlit as st
import joblib
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
model = joblib.load(BASE_DIR / "california_house_price_model.pkl")



st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.sidebar.title("🏠 California Housing")

st.sidebar.markdown("---")

st.sidebar.info("""
This AI model predicts California House Prices
using Multiple Linear Regression.
""")

st.sidebar.success("Model Loaded Successfully ✅")

st.title("🏠 California House Price Predictor")

st.caption("Predict California House Prices using Machine Learning")

st.markdown("---")

st.header("Enter House Details")

income = st.number_input("Median Income", min_value=0.0)

house_age = st.number_input("House Age", min_value=0.0)

avg_rooms = st.number_input("Average Rooms", min_value=0.0)

avg_bedrooms = st.number_input("Average Bedrooms", min_value=0.0)

population = st.number_input("Population", min_value=0.0)

avg_occupancy = st.number_input("Average Occupancy", min_value=0.0)

latitude = st.number_input("Latitude")

longitude = st.number_input("Longitude")
if st.button("🔮 Predict Price"):

    features = np.array([[
        income,
        house_age,
        avg_rooms,
        avg_bedrooms,
        population,
        avg_occupancy,
        latitude,
        longitude
    ]])
    st.markdown("---")

    prediction = model.predict(features)

    st.metric(
    label="Predicted House Price",
    value=f"${prediction[0]*100000:,.2f}"
)
