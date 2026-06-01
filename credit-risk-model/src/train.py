import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn

# def load_and_split_data(data_path, target_col='is_high_risk', test_size=0.2, random_state=42):
#     """Loads processed dataset and splits it into training and testing sets."""
#     if not os.path.exists(data_path):
#         raise FileNotFoundError(f"Processed dataset not found at {data_path}")
        
#     df = pd.read_csv(data_path)
    
#     # Drop identifier columns not used for training weights
#     X = df.drop(columns=[target_col, 'CustomerId'], errors='ignore')
#     y = df[target_col]
    
#     # Split data while stratifying to preserve the tiny minority risk class distribution
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, random_state=random_state, stratify=y
#     )
#     return X_train, X_test, y_train, y_test


def load_and_split_data(data_path, target_col='is_high_risk', test_size=0.2, random_state=42):
    """Loads processed dataset and splits it into training and testing sets with a safety fall-back for low minority classes."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found at {data_path}")
        
    df = pd.read_csv(data_path)
    
    # Drop identifier columns not used for training weights
    X = df.drop(columns=[target_col, 'CustomerId'], errors='ignore')
    y = df[target_col]
    
    # Count members in the minority group
    minority_class_count = y.value_counts().min()
    
    if minority_class_count < 2:
        print(f"⚠️ Warning: The minority class has only {minority_class_count} member(s).")
        print("Bypassing stratification split to avoid scikit-learn errors...")
        # Fall back to standard random split without stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    else:
        # Standard stratified split when class sizes permit it safely
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
    return X_train, X_test, y_train, y_test

def evaluate_model(y_true, y_pred, y_prob):
    """Computes critical evaluation metrics for classification performance."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        # Handle cases where ROC-AUC cannot be calculated if prediction values are uniform
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    }
    return metrics

def train_and_track_experiments():
    """Trains classification models and records all activities to MLflow."""
    processed_data_path = '../data/processed/processed_data.csv'
    
    # Handle absolute path fallback if run from repo root folder context
    if not os.path.exists(processed_data_path):
        processed_data_path = 'data/processed/processed_data.csv'
        
    print("Loading data and creating stratified train/test splits...")
    X_train, X_test, y_train, y_test = load_and_split_data(processed_data_path)
    
    # Set the MLflow tracking experiment name
    mlflow.set_experiment("Bati_Bank_Credit_Risk_Modeling")
    
    # ------------------ EXPERIMENT 1: Logistic Regression ------------------
    print("\n--- Starting Experiment 1: Logistic Regression ---")
    with mlflow.start_run(run_name="Logistic_Regression_Baseline"):
        # Define model parameters
        lr_params = {"C": 1.0, "max_iter": 1000, "class_weight": "balanced", "random_state": 42}
        lr_model = LogisticRegression(**lr_params)
        
        # Train model
        lr_model.fit(X_train, y_train)
        
        # Run inference predictions
        y_pred = lr_model.predict(X_test)
        y_prob = lr_model.predict_proba(X_test)[:, 1]
        
        # Score metrics
        metrics = evaluate_model(y_test, y_pred, y_prob)
        print(f"Logistic Regression F1-Score: {metrics['f1_score']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
        
        # Log to MLflow dashboard
        mlflow.log_params(lr_params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(lr_model, "model", registered_model_name="LogisticRegression_Risk_Model")
        print("Logged Logistic Regression configurations to MLflow successfully.")

    # ------------------ EXPERIMENT 2: Random Forest ------------------
    print("\n--- Starting Experiment 2: Random Forest ---")
    with mlflow.start_run(run_name="Random_Forest_Classifier"):
        # Define model parameters
        rf_params = {"n_estimators": 100, "max_depth": 5, "class_weight": "balanced", "random_state": 42}
        rf_model = RandomForestClassifier(**rf_params)
        
        # Train model
        rf_model.fit(X_train, y_train)
        
        # Run inference predictions
        y_pred = rf_model.predict(X_test)
        y_prob = rf_model.predict_proba(X_test)[:, 1]
        
        # Score metrics
        metrics = evaluate_model(y_test, y_pred, y_prob)
        print(f"Random Forest F1-Score: {metrics['f1_score']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
        
        # Log to MLflow dashboard
        mlflow.log_params(rf_params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(rf_model, "model", registered_model_name="RandomForest_Risk_Model")
        print("Logged Random Forest configurations to MLflow successfully.")

if __name__ == "__main__":
    train_and_track_experiments()