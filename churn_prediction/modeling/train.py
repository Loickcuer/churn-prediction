import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
import mlflow
import mlflow.sklearn
import joblib
from config import MODELS_FOLDER, MLFLOW_EXPERIMENT_NAME, PROCESSED_DATA_PATH

def split_data(df, target_col='churn'): # Divise les données en ensembles d'entraînement et de test.
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def handle_class_imbalance(X_train, y_train, method='smote'): # Gère le déséquilibre de classe en utilisant différentes techniques.
    if method == 'smote':
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    elif method == 'smoteenn':
        smoteenn = SMOTEENN(random_state=42)
        X_resampled, y_resampled = smoteenn.fit_resample(X_train, y_train)
    elif method == 'undersampling':
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    else:
        X_resampled, y_resampled = X_train, y_train
    
    return X_resampled, y_resampled

def train_random_forest(X_train, y_train): # Entraîne un modèle de forêt aléatoire.
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestClassifier(random_state=42)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    random_search = RandomizedSearchCV(rf, param_distributions=param_grid, n_iter=20, cv=cv, scoring='f1', random_state=42, n_jobs=-1)
    
    random_search.fit(X_train, y_train)
    
    return random_search.best_estimator_

def train_logistic_regression(X_train, y_train): # Entraîne un modèle de régression logistique.
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
    
    lr = LogisticRegression(random_state=42)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    random_search = RandomizedSearchCV(lr, param_distributions=param_grid, n_iter=20, cv=cv, scoring='f1', random_state=42, n_jobs=-1)
    
    random_search.fit(X_train, y_train)
    
    return random_search.best_estimator_

def train_xgboost(X_train, y_train): # Entraîne un modèle XGBoost.
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    
    xgb = XGBClassifier(random_state=42)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    random_search = RandomizedSearchCV(xgb, param_distributions=param_grid, n_iter=20, cv=cv, scoring='f1', random_state=42, n_jobs=-1)
    
    random_search.fit(X_train, y_train)
    
    return random_search.best_estimator_

def evaluate_model(model, X_test, y_test): # Évalue les performances du modèle en utilisant différentes métriques.
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }

def save_model(model, model_name): # Sauvegarde le modèle au format joblib.
    joblib.dump(model, f"{MODELS_FOLDER}/{model_name}.joblib")

def train_and_evaluate(model_type, X_train, y_train, X_test, y_test): # Entraîne et évalue un modèle donné.
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run():
        if model_type == "random_forest":
            model = train_random_forest(X_train, y_train)
        elif model_type == "logistic_regression":
            model = train_logistic_regression(X_train, y_train)
        elif model_type == "xgboost":
            model = train_xgboost(X_train, y_train)
        else:
            raise ValueError("Type de modèle non reconnu")
        
        metrics = evaluate_model(model, X_test, y_test)
        
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, model_type)
        
        save_model(model, f"best_{model_type}")
        
        print(f"Métriques pour {model_type}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return model, metrics