import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
from config import PROCESSED_DATA_PATH, MODELS_FOLDER, XGB_IMPORTANT_FEATURES
import joblib
import os

def load_processed_data():
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    X = df[XGB_IMPORTANT_FEATURES]
    y = df['churn']
    return X, y

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def handle_class_imbalance(X_train, y_train):
    rus = RandomUnderSampler(sampling_strategy='majority', random_state=42)
    return rus.fit_resample(X_train, y_train)

def train_xgboost(X_train, y_train):
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 2, 3, 4, 5],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4]
    }
    
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_random = RandomizedSearchCV(estimator=xgb, param_distributions=param_dist, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
    xgb_random.fit(X_train, y_train)
    
    return xgb_random.best_estimator_, xgb_random.best_params_

def evaluate_model(model, X_train, y_train, X_test, y_test):
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, pred_train)
    test_accuracy = accuracy_score(y_test, pred_test)
    precision = precision_score(y_test, pred_test, average='weighted')
    recall = recall_score(y_test, pred_test, average='weighted')
    f1 = f1_score(y_test, pred_test, average='weighted')
    
    return {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def save_model(model, model_name):
    joblib.dump(model, os.path.join(MODELS_FOLDER, f"{model_name}.joblib"))