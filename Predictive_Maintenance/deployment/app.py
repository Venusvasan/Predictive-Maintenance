import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_model.joblib")
model = joblib.load(model_path)

st.title("Engine Health Predictor")
st.write("""
 Predict and classify whether an engine requires maintenance or is operating normally using real time engine sensor data
""")
# Organize input form into 3 clear columns
col1, col2 = st.columns(2)

with col1:
    engine_rpm = st.number_input("Engine RPM", min_value=0, max_value=5000, value=750)
    lub_oil_pressure = st.number_input("Lub Oil Pressure(bar)", min_value=0, max_value=10, value=4)
    fuel_pressure = st.number_input("Fuel Pressure(bar)", min_value=0, max_value=25, value=6)

with col2:
    coolant_pressure = st.number_input("Coolant Pressure(bar)", min_value=0.0, max_value=10.0, value=2.0)
    lub_oil_temp = st.number_input("Lub Oil Temperature(Celsius)", min_value=60.0, max_value=110.0, value=70.0)
    coolant_temp = st.number_input("Coolant Temperature(Celsius)", min_value=50.0, max_value=210.0, value=80.0)

input_data = pd.DataFrame([{
    "Engine rpm": engine_rpm,
    "Lub oil pressure": lub_oil_pressure,
    "Fuel pressure": fuel_pressure,
    "Coolant pressure": coolant_pressure,
    "lub oil temp": lub_oil_temp,
    "Coolant temp": coolant_temp
}])

st.divider()

if st.button("Predict Engine Condition"):
    threshold = 0.55
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    pred_prob = probabilities[1]
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(
            f"### Needs Maintenance!\n"
            f"**Likelihood to Breakdown: {pred_prob:.1%}**\n\n"
            f"Recommendation: Prioritize this lead for direct marketing contact."
        )
        st.balloons()
    else:
        st.info(
            f"### Engine running normally\n"
            f"**Likelihood to Breakdown: {pred_prob:.1%}**\n\n"
            f"Recommendation: Deprioritize to preserve sales resources."
        )
