import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import mlflow
import subprocess
import os
from config import PROCESSED_DATA_PATH, XGB_IMPORTANT_FEATURES

def load_model():
    model_path = os.path.join("..", "models", "XGBoost_Tuned_RUS_Important_Features.joblib")
    return joblib.load(model_path)

def make_predictions(model, X_test):
    return model.predict(X_test)

def make_probability_predictions(model, X_test):
    return model.predict_proba(X_test)[:, 1]

def evaluate_model(y_true, y_pred, y_pred_proba):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {auc_score:.4f}")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc_score
    }

def predict_and_evaluate(X_test, y_test):
    model = load_model()
    y_pred = make_predictions(model, X_test)
    y_pred_proba = make_probability_predictions(model, X_test)
    metrics = evaluate_model(y_test, y_pred, y_pred_proba)
    
    return y_pred, y_pred_proba, metrics

def save_predictions_to_excel(df, y_pred, y_pred_proba):
    df['churn_score'] = y_pred_proba
    df['churn_class'] = pd.cut(df['churn_score'] * 100, bins=[-1, 20, 40, 60, 80, 100], labels=["Very Low", "Low", "Average", "High Risk", "Very High Risk"])
    df.to_excel("../docs/df.xlsx", index=False)
    print("Predictions saved to ../docs/df.xlsx")
    print("\nSample of predictions:")
    print(df[['churn_score', 'churn_class']].head())
    
    # Create and save df_high_risk
    df_high_risk = df[df['churn_class'] == 'Very High Risk']
    print("\nHigh risk clients:")
    print(df_high_risk.head())
    df_high_risk.to_excel('../docs/df_high_risk.xlsx', index=False)
    print("High risk clients saved to ../docs/df_high_risk.xlsx")

def run_streamlit_app():
    streamlit_path = os.path.join("..", "docs", "streamlit_app.py")
    subprocess.Popen(["streamlit", "run", streamlit_path])