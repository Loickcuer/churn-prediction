import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pyod.models.iforest import IForest
from config import DATA_PATH, INTERIM_DATA_PATH

def load_data(path=DATA_PATH): # Charge les données à partir du fichier CSV.
    return pd.read_csv(path, delimiter=';')

def preprocess_data(df): # Prétraite les données en effectuant les opérations suivantes:
    # Conversion des types
    df['interet_compte_epargne_total'] = pd.to_numeric(df['interet_compte_epargne_total'], errors='coerce')
    df['id_client'] = df['id_client'].astype(str)

    # Gestion des valeurs manquantes
    catcols = [col for col in df.columns if df[col].dtype == 'object']
    for col in catcols:
        df[col] = df[col].fillna(df[col].mode()[0])

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Conversion des types booléens
    bool_columns = ['espace_client_web', 'assurance_vie', 'banque_principale', 
                    'compte_epargne', 'compte_courant', 'churn', 'compte_titres']
    for col in bool_columns:
        df[col] = df[col].map({'oui': True, 'non': False})

    # Traitement des valeurs aberrantes
    df = remove_outliers(df)

    return df

def remove_outliers(df): #Supprime les valeurs aberrantes en utilisant Isolation Forest.
    
    clf = IForest(contamination=0.001, random_state=42)
    
    for col in ['anciennete_mois', 'interet_compte_epargne_total']:
        values = df[[col]].values
        clf.fit(values)
        y_pred = clf.predict(values)
        df = df[y_pred == 0]
    
    return df

def save_processed_data(df, path=INTERIM_DATA_PATH): # Sauvegarde les données prétraitées au format Parquet.

    df.to_parquet(path, engine='auto', compression='snappy', index=False)

def load_processed_data(path=INTERIM_DATA_PATH):    # Charge les données prétraitées à partir du fichier Parquet.
    return pd.read_parquet(path)