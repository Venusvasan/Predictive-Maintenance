import pandas as pd
import warnings
RAW_PATH = "Predictive_Maintenance/Data/engine_data.csv"

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# Validate that the expected columns are present before registering it
expected_columns = ["Engine rpm", "Lub oil pressure", "Fuel pressure", "Coolant pressure", "lub oil temp", "Coolant temp",
    "Engine Condition"]
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

# Check for null / missing values
null_count = df.isnull().sum().sum()
if null_count > 0:
    warnings.warn(f"Warning: Dataset contains {null_count} missing values.")
else:
    print("No missing values detected in the dataset.")

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("Target Variable Distribution (Engine Condition):")
print(df['Engine Condition'].value_counts(normalize=True).rename({0: "0 (Off/False/Active)", 1: "1 (On/True/Faulty)"}))
