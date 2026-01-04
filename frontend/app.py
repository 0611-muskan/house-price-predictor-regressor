import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('house_price_model.pkl')

st.set_page_config(page_title="🏡 House Price Prediction", layout="centered")

st.title("🏠 House Price Prediction App")
st.markdown("""
Use this app to predict the estimated **price of a house** based on various features.  
Enter the property details below and click **Predict Price**.
""")

# Sidebar for inputs
st.sidebar.header("🧾 Enter Property Details")

# Collect user input for California Housing dataset features
MedInc = st.sidebar.number_input("Median Income (in $10,000s)", min_value=0.0, max_value=15.0, value=3.0, step=0.1)
HouseAge = st.sidebar.number_input("House Age (years)", min_value=1, max_value=52, value=28)
AveRooms = st.sidebar.number_input("Average Rooms", min_value=1.0, max_value=10.0, value=5.0, step=0.1)
AveBedrms = st.sidebar.number_input("Average Bedrooms", min_value=0.5, max_value=5.0, value=1.1, step=0.1)
Population = st.sidebar.number_input("Population", min_value=100, max_value=35000, value=1500, step=100)
AveOccup = st.sidebar.number_input("Average Occupancy", min_value=1.0, max_value=10.0, value=3.0, step=0.1)
Latitude = st.sidebar.number_input("Latitude", min_value=32.0, max_value=42.0, value=37.88, step=0.01)
Longitude = st.sidebar.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-122.23, step=0.01)

# Convert user input into DataFrame
input_data = pd.DataFrame({
    'MedInc': [MedInc],
    'HouseAge': [HouseAge],
    'AveRooms': [AveRooms],
    'AveBedrms': [AveBedrms],
    'Population': [Population],
    'AveOccup': [AveOccup],
    'Latitude': [Latitude],
    'Longitude': [Longitude]
})

# Predict
if st.sidebar.button("🔮 Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"🏡 **Estimated House Price:** ${prediction * 100000:,.2f}")
    st.balloons()

st.markdown("---")
st.caption("Made with ❤️ by Muskan,Kanan, Aditya, Samridhi — MCA (2025–2027)")
