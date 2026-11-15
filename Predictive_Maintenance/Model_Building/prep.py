
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "Predictive_Maintenance/Data/engine_data.csv"

df = pd.read_csv(DATA_PATH)

# Drop identifier columns not useful for prediction
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Separate features and target
X = df.drop(columns=["Engine Condition"])
y = df["Engine Condition"]

# Perform Train/Test split (80/20) with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Missing values treatment (using median of predictor variables on the train set)
if X_train.isnull().any().any():
    print("Missing values detected in predictors. Applying median imputation...")
    # Calculate median on train data only to prevent data leakage
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)
else:
    print("No missing values detected in predictor variables.")

X_train.to_csv("Xtrain.csv", index=False)
X_test.to_csv("Xtest.csv", index=False)
y_train.to_csv("ytrain.csv", index=False)
y_test.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print(f"the shape of the train and test splits are {X_train.shape} and {X_test.shape}")
