import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
import subprocess

from config import DATA_PATH, INTERIM_DATA_PATH, PROCESSED_DATA_PATH, MODELS_FOLDER, XGB_IMPORTANT_FEATURES
from features import load_data, preprocess_data, save_interim_data, load_interim_data, create_features, one_hot_encoding, save_processed_data, load_processed_data
from modeling.train import split_data, handle_class_imbalance, train_xgboost, evaluate_model, save_model
from modeling.predict import make_predictions, make_probability_predictions, evaluate_model as predict_evaluate_model, save_predictions_to_excel, run_streamlit_app

def main():
    print("Chargement et prétraitement des données...")
    df = load_data(DATA_PATH)
    df = preprocess_data(df)
    save_interim_data(df, INTERIM_DATA_PATH)

    print("Création des features...") 
    df = load_interim_data(INTERIM_DATA_PATH)
    df = create_features(df)
    df = one_hot_encoding(df)
    save_processed_data(df, PROCESSED_DATA_PATH)

    print("Préparation des données pour l'entraînement...")
    df = load_processed_data(PROCESSED_DATA_PATH)
    X = df[XGB_IMPORTANT_FEATURES]
    y = df['churn']
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_resampled, y_train_resampled = handle_class_imbalance(X_train, y_train)

    print("Entraînement du modèle...")
    best_model, best_params = train_xgboost(X_train_resampled, y_train_resampled)
    
    print("Évaluation du modèle...")
    metrics = evaluate_model(best_model, X_train, y_train, X_test, y_test)
    print("Métriques d'évaluation:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    model_name = "XGBoost_Tuned_RUS_Important_Features"
    save_model(best_model, model_name)
    print(f"Modèle sauvegardé sous le nom: {model_name}")

    print("Prédictions et évaluation finale...")
    final_model = joblib.load(os.path.join(MODELS_FOLDER, f"{model_name}.joblib"))
    y_pred = make_predictions(final_model, X_test)
    y_pred_proba = make_probability_predictions(final_model, X_test)
    final_metrics = predict_evaluate_model(y_test, y_pred, y_pred_proba)

    print("Sauvegarde des prédictions...")
    save_predictions_to_excel(df.loc[X_test.index], y_pred, y_pred_proba)

    print("Lancement de l'application Streamlit...")
    run_streamlit_app()

if __name__ == "__main__":
    main()