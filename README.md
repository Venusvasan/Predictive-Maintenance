# Predictive Maintenance for Automotive & Industrial Engines

An end-to-end Machine Learning and MLOps pipeline that predicts engine maintenance requirements from onboard sensor data (`Engine rpm`, `Lub oil pressure`, `Fuel pressure`, `Coolant pressure`, `lub oil temp`, and `Coolant temp`).

Built with **Scikit-Learn**, **XGBoost**, **MLflow**, **GitHub Actions CI/CD**, and **Streamlit**.

---

## Problem Statement & Business Context

Unplanned engine breakdowns cause costly repairs, operational downtime, and safety hazards for vehicle owners and commercial fleet operators. This project classifies engine health into two states:
- **`0` (Normal / Healthy):** Engine operating within safe parameters.
- **`1` (Faulty / Maintenance Required):** Engine operating with faulty conditions requiring servicing.

---

## Repository Architecture

```text
Predictive-Maintenance/
├── .github/
│   └── workflows/
│       └── pipeline.yml                  # Automated CI/CD MLOps workflow
├── Predictive_Maintenance/
│   ├── Data/
│   │   └── engine_data.csv               # Raw telemetry dataset (19,535 rows, 7 columns)
│   ├── Model_Building/
│   │   ├── data_register.py              # Schema & null-value validation gatekeeper
│   │   ├── prep.py                       # Stratified 80/20 train-test split exporter
│   │   └── train.py                      # GridSearchCV tuning, MLflow tracking & model export
│   ├── deployment/
│   │   ├── app.py                        # Streamlit web application for real-time inference
│   │   ├── best_model.joblib             # Serialized winning XGBoost pipeline
│   │   └── requirements.txt              # Streamlit Cloud dependencies
│   └── requirements.txt                  # CI/CD pipeline dependencies
└── README.md
