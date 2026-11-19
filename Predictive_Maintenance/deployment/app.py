import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_model.joblib")
model = joblib.load(model_path)

# Page configuration
st.set_page_config(
    page_title="Engine Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean cards and metrics
st.markdown("""
    <style>
    .hero-banner {
        background: linear-gradient(135deg, #1B365D 0%, #00688B 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .hero-sub {
        font-size: 1rem;
        color: #E0EEF5;
        margin-bottom: 0.75rem;
    }
    .meta-badge {
        display: inline-block;
        background-color: rgba(255, 255, 255, 0.16);
        border: 1px solid rgba(255, 255, 255, 0.35);
        padding: 0.25rem 0.75rem;
        border-radius: 10px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .section-card {
        background-color: #F4F8FB;
        border-left: 4px solid #00688B;
        padding: 0.6rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.8rem;
        font-weight: 600;
        color: #1B365D;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">⚙️ Engine Predictive Maintenance System</div>
        <div class="hero-sub">Real-time Engine Health detection and failure risk scoring using onboard engine sensor data.</div>
        <span class="meta-badge">Algorithm: XGBoost Classifier</span>
        <span class="meta-badge">Decision Threshold: 0.55</span>
    </div>
""", unsafe_allow_html=True)

# Input Form Layout
st.subheader("Enter Sensor data")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="section-card">🔄 Speed Sensor</div>', unsafe_allow_html=True)    
    engine_rpm = st.number_input("Engine RPM", min_value=50, max_value=2500, value=900, step=10, help="Crankshaft speed in revolutions per minute")
with col2:
    st.markdown('<div class="section-card">⏲️ Pressure Sensors </div>', unsafe_allow_html=True)    
    fuel_pressure = st.number_input("Fuel Pressure (bar)", min_value=0.0, max_value=25.0, value=5.0, step=0.1, help="Fuel injection supply pressure")
    coolant_pressure = st.number_input("Coolant Pressure (bar)", min_value=0.0, max_value=10.0, value=2.0, step=0.1, help="Liquid cooling circuit pressure")
    lub_oil_pressure = st.number_input("Lub Oil Pressure (bar)", min_value=0.0, max_value=10.0, value=4.0, step=0.1, help="Lubricating oil circuit pressure")

with col3:
    st.markdown('<div class="section-card">🌡️ Thermal Sensors</div>', unsafe_allow_html=True)
    lub_oil_temp = st.number_input("Lub Oil Temp (°C)", min_value=60.0, max_value=110.0, value=70.0, step=0.5, help="Operating temperature of lubricating oil")
    coolant_temp = st.number_input("Coolant Temp (°C)", min_value=50.0, max_value=210.0, value=75.0, step=0.5, help="Engine coolant temperature")

# Construct DataFrame matching exact training column names
input_df = pd.DataFrame([{
    "Engine rpm": engine_rpm,
    "Lub oil pressure": lub_oil_pressure,
    "Fuel pressure": fuel_pressure,
    "Coolant pressure": coolant_pressure,
    "lub oil temp": lub_oil_temp,
    "Coolant temp": coolant_temp
}])

with st.expander("View Formatted Data Payload (DataFrame)", expanded=False):
    st.dataframe(input_df, use_container_width=True)

st.divider()

# Inference Section
if st.button("🔍 Run Diagnostic Prediction", type="primary", use_container_width=True):
    fault_prob = float(model.predict_proba(input_df)[0][1])
    threshold = 0.55
    prediction = int(fault_prob >= threshold)

    st.subheader("2. Diagnostic Assessment Result")
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.metric(
            label="Predicted Failure Probability",
            value=f"{fault_prob:.2%}",
            delta=f"{(fault_prob - threshold):+.2%} vs Threshold",
            delta_color="inverse"
        )
        st.progress(min(max(fault_prob, 0.0), 1.0))

    with res_col2:
        if fault_prob >= 0.65:
            st.error(
                f"### 🚨 CRITICAL FAULT RISK (Class 1 — Maintenance Required)\n"
                f"Failure probability (**{fault_prob:.2%}**) exceeds the critical threshold (`0.65`).\n\n"
                f"**Action:** Halt equipment immediately and schedule priority diagnostic inspection."
            )
        elif prediction == 1:
            st.warning(
                f"### ⚠️ ADVISORY WARNING (Class 1 — Maintenance Required)\n"
                f"Failure probability (**{fault_prob:.2%}**) exceeds the operating threshold (`{threshold:.2f}`).\n\n"
                f"**Action:** Schedule preventive  inspection at the earliest"
            )
        else:
            st.success(
                f"### ✅ NORMAL OPERATION (Class 0 — Healthy)\n"
                f"Failure probability (**{fault_prob:.2%}**) is below the operating threshold (`{threshold:.2f}`).\n\n"
                f"**Action:** Engine parameters are within safe limits. Continue routine schedule."
            )
