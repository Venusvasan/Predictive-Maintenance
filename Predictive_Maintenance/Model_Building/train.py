import os
import joblib
import pandas as pd
import mlflow
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

print("--- Starting Model Building and Tracking ---")

# 1. Initialize MLflow
os.makedirs("mlruns", exist_ok=True)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Engine_Predictive_Maintenance")

# 2. Load dataset splits
X_train = pd.read_csv("Xtrain.csv")
X_test = pd.read_csv("Xtest.csv")
y_train = pd.read_csv("ytrain.csv").squeeze()
y_test = pd.read_csv("ytest.csv").squeeze()

# Identify numeric sensor features
numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()

# Shared preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features)
)

# 3. Model definitions
rf_pipeline = make_pipeline(
    preprocessor,
    RandomForestClassifier(random_state=42, class_weight="balanced")
)
xgb_pipeline = make_pipeline(
    preprocessor,
    xgb.XGBClassifier(random_state=42, eval_metric="logloss")
)

# 4. Parameter Grids
rf_param_grid = {
    "randomforestclassifier__n_estimators": [50, 100],
    "randomforestclassifier__max_depth": [5, 10, None],
    "randomforestclassifier__min_samples_split": [2, 5]
}

xgb_param_grid = {
    "xgbclassifier__n_estimators": [50,100],
    "xgbclassifier__max_depth": [3, 4],
    "xgbclassifier__colsample_bytree": [0.4, 0.5, 0.6],
    "xgbclassifier__colsample_bylevel": [0.4, 0.5, 0.6],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__reg_lambda": [0.4, 0.5, 0.6],
}
model_config = {
    "RandomForest": {"pipeline": rf_pipeline, "params": rf_param_grid},
    "XGBoost": {"pipeline": xgb_pipeline, "params": xgb_param_grid}
}

best_f1 = -1.0
best_model = None
best_name = ""
classification_threshold = 0.55

# 5. Hyper tuning
for model_name, config in model_config.items():
    print(f"\n==========================================")
    print(f"Training & Tuning: {model_name}")
    print(f"==========================================")

    with mlflow.start_run(run_name=model_name):
        grid_search = GridSearchCV(
            estimator=config["pipeline"],
            param_grid=config["params"],
            cv=3,
            scoring="f1",
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        best_estimator = grid_search.best_estimator_


        y_pred_train_proba = best_estimator.predict_proba(X_train)[:, 1]
        y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

        y_pred_test_proba = best_estimator.predict_proba(X_test)[:, 1]
        y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

        train_report = classification_report(y_train, y_pred_train, output_dict=True)
        test_report = classification_report(y_test, y_pred_test, output_dict=True)
        test_auc = roc_auc_score(y_test, y_pred_test_proba)

        # Log to MLflow
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics({
            "train_accuracy": train_report["accuracy"],
            "train_precision": train_report["1"]["precision"],
            "train_recall": train_report["1"]["recall"],
            "train_f1-score": train_report["1"]["f1-score"],
            "test_accuracy": test_report["accuracy"],
            "test_precision": test_report["1"]["precision"],
            "test_recall": test_report["1"]["recall"],
            "test_f1-score": test_report["1"]["f1-score"],
            "test_roc_auc": test_auc
        })

        print(f"Best Params for {model_name}: {grid_search.best_params_}")
        #print(f"Test Recall: {test_report['1']['recall']:.4f} | Test F1: {test_report['1']['f1-score']:.4f} | Test AUC: {test_auc:.4f}")
        print(classification_report(y_test, y_pred_test))

        # Check if this model outperforms previous candidates
        model_f1 = test_report["1"]["f1-score"]
        if model_f1 > best_f1:
            best_f1 = model_f1
            best_model = best_estimator
            best_name = model_name

# 6. Save the winning model for Streamlit and GitHub Actions
os.makedirs("Predictive_Maintenance/deployment", exist_ok=True)
model_path = "Predictive_Maintenance/deployment/best_model.joblib"
joblib.dump(best_model, model_path)
mlflow.log_artifact(model_path, artifact_path="model")

print("\n**************************************************")
print(f"Best Model Selected: {best_name} (Test F1: {best_f1:.4f})")
print(f"Model saved to {model_path}")
print("**************************************************")
