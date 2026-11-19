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
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1B365D;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)




# Sidebar: Quick Test Presets & Model Metadata
with st.sidebar:
    st.header("⚙️ System Controls")
    preset = st.radio(
        "Load Sample Sensor Profile:",
        ["Custom Input", "Normal Operating Engine", "High-Risk Faulty Engine"],
        index=0
    )
    st.divider()
    st.subheader("Model Specifications")
    st.markdown("""
    - **Algorithm:** XGBoost Classifier
    - **Decision Threshold:** `0.55`
    """)
# Default values based on preset
if preset == "Normal Operating Engine":
    def_rpm, def_oil_p, def_fuel_p = 920, 3.80, 5.40
    def_cool_p, def_oil_t, def_cool_t = 2.10, 76.50, 77.50
elif preset == "High-Risk Faulty Engine":
    def_rpm, def_oil_p, def_fuel_p = 540, 2.10, 12.80
    def_cool_p, def_oil_t, def_cool_t = 3.60, 84.50, 86.00
else:
    def_rpm, def_oil_p, def_fuel_p = 746, 3.16, 6.20
    def_cool_p, def_oil_t, def_cool_t = 2.17, 76.82, 78.35

# Header
st.markdown('<div class="main-header">⚙️ Engine Predictive Maintenance System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Real-time anomaly detection and failure risk scoring using onboard engine sensor telemetry.</div>',
    unsafe_allow_html=True
)

# Input Form Layout
st.subheader("Enter Sensor data")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Speed Sensor")
    engine_rpm = st.number_input("Engine RPM", min_value=50, max_value=2500, value=def_rpm, step=10, help="Crankshaft speed in revolutions per minute")
with col2:
    st.markdown("**Pressures Sensors**")
    fuel_pressure = st.number_input("Fuel Pressure (bar)", min_value=0.0, max_value=25.0, value=def_fuel_p, step=0.1, help="Fuel injection supply pressure")
    coolant_pressure = st.number_input("Coolant Pressure (bar)", min_value=0.0, max_value=10.0, value=def_cool_p, step=0.1, help="Liquid cooling circuit pressure")
    lub_oil_pressure = st.number_input("Lub Oil Pressure (bar)", min_value=0.0, max_value=10.0, value=def_oil_p, step=0.1, help="Lubricating oil circuit pressure")

with col3:
    st.markdown("**Thermal Sensors**")
    lub_oil_temp = st.number_input("Lub Oil Temp (°C)", min_value=60.0, max_value=110.0, value=def_oil_t, step=0.5, help="Operating temperature of lubricating oil")
    coolant_temp = st.number_input("Coolant Temp (°C)", min_value=50.0, max_value=210.0, value=def_cool_t, step=0.5, help="Engine coolant temperature")

# Construct DataFrame matching exact training column names
input_df = pd.DataFrame([{
    "Engine rpm": engine_rpm,
    "Lub oil pressure": lub_oil_pressure,
    "Fuel pressure": fuel_pressure,
    "Coolant pressure": coolant_pressure,
    "lub oil temp": lub_oil_temp,
    "Coolant temp": coolant_temp
}])

with st.expander("View Formatted Telemetry Payload (DataFrame)", expanded=False):
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
                f"**Action:** Schedule preventive workshop inspection at next depot return."
            )
        else:
            st.success(
                f"### ✅ NORMAL OPERATION (Class 0 — Healthy)\n"
                f"Failure probability (**{fault_prob:.2%}**) is below the operating threshold (`{threshold:.2f}`).\n\n"
                f"**Action:** Engine parameters are within safe limits. Continue routine schedule."
            )
