import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pyod.models.iforest import IForest
import os
from config import DATA_PATH, INTERIM_DATA_PATH, PROCESSED_DATA_PATH

def load_data(path=DATA_PATH):
    return pd.read_csv(path, delimiter=';')

def preprocess_data(df):
    # Conversion des types
    df['interet_compte_epargne_total'] = pd.to_numeric(df['interet_compte_epargne_total'], errors='coerce')
    df['id_client'] = df['id_client'].astype(str)

    # Gestion des valeurs manquantes
    catcols = df.select_dtypes(include=['object']).columns
    for col in catcols:
        df[col] = df[col].fillna(df[col].mode()[0])

    numcols = df.select_dtypes(include=[np.number]).columns
    for col in numcols:
        df[col] = df[col].fillna(df[col].median())

    # Conversion des types booléens
    bool_columns = ['espace_client_web', 'assurance_vie', 'banque_principale', 
                    'compte_epargne', 'compte_courant', 'churn', 'compte_titres']
    for col in bool_columns:
        df[col] = df[col].map({'oui': True, 'non': False})

    # Traitement des valeurs aberrantes
    df = remove_outliers(df)

    return df

def remove_outliers(df):
    clf = IForest(contamination=0.001, random_state=42)
    
    for col in ['anciennete_mois', 'interet_compte_epargne_total']:
        values = df[[col]].values
        clf.fit(values)
        y_pred = clf.predict(values)
        df = df[y_pred == 0]
    
    return df

def save_interim_data(df, path = INTERIM_DATA_PATH):
    df.to_parquet(path, engine='auto', compression='snappy', index=False)

def load_interim_data(path = PROCESSED_DATA_PATH):
    return pd.read_parquet(path)

def save_processed_data(df, path = INTERIM_DATA_PATH):
    df.to_parquet(path, engine='auto', compression='snappy', index=False)

def load_processed_data(path = PROCESSED_DATA_PATH):
    return pd.read_parquet(path)

def calculer_score_engagement(row):
    score = 0
    if row['espace_client'] == "oui":
        score += 30
    elif row['espace_client'] == "inconnu":
        score += 10
    elif row['espace_client'] == "non":
        score += 0
    
    if row['espace_client_web'] == True:
        score += 20
    
    score += min(row['anciennete_mois'] / 12 * 5, 50)
    
    produits = ['compte_epargne', 'compte_titres', 'assurance_vie', 'PEA_oui', 
                'assurance_auto_oui', 'assurance_habitation_oui', 'credit_immo_oui', 
                'credit_autres_bail', 'credit_autres_consommation', 'credit_autres_permanent']
    score += sum(row[produit] for produit in produits if produit in row) * 10
    
    if row['cartes_bancaires'] == "premium":
        score += 20
    elif row['cartes_bancaires'] == "medium":
        score += 15
    elif row['cartes_bancaires'] == "basic":
        score += 10
    
    if row['banque_principale'] == True:
        score += 20
    
    return score

def calculer_score_risque_financier(df):
    df['score_risque_financier'] = 0
    df['score_risque_financier'] += df['agios_6mois'] / df['agios_6mois'].max() * 30
    df['score_risque_financier'] += df['credit_autres'].map({'bail': 10, 'consommation': 15, 'permanent': 20}).fillna(0)
    df['score_risque_financier'] += np.maximum(0, 20 - df['anciennete_mois'] / 12)
    df['score_risque_financier'] -= ((df['compte_epargne'].astype(int) + df['compte_titres'].astype(int) + (df['PEA'] == 'oui').astype(int)) * 5 +df['assurance_vie'].astype(int) * 10)
    df.loc[df['credit_immo'] == 'oui','score_risque_financier'] -= 10
    df['score_risque_financier'] = np.clip(df['score_risque_financier'], 0, None)
    max_score = df['score_risque_financier'].max()
    df['score_risque_financier'] = (df['score_risque_financier'] / max_score) * 100
    df['score_risque_financier'] = df['score_risque_financier'].round(2)
    return df

def creer_categorie_age(df):
    df['categorie_age'] = pd.cut(df['age'], bins=[0, 18, 25, 35, 45, 55, 65, float('inf')], labels=['Mineur', '18-24', '25-34', '35-44', '45-54', '55-64', '65+'], right=False)
    return df

def apply_pca(df, columns_to_pca):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[columns_to_pca])
    
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    cum_var_ratio = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cum_var_ratio >= 0.82) + 1
    
    pca_final = PCA(n_components=n_components)
    X_pca_final = pca_final.fit_transform(X_scaled)
    
    new_columns = [f'PC{i+1}' for i in range(n_components)]
    df = df.drop(columns_to_pca, axis=1)
    df[new_columns] = X_pca_final
    
    return df

def create_features(df):
    
    columns_to_pca = [col for col in df.columns if col.startswith('var_')]
    df = apply_pca(df, columns_to_pca)
    
    df['score_engagement'] = df.apply(calculer_score_engagement, axis=1)
    df = calculer_score_risque_financier(df)
    df = creer_categorie_age(df)
    
    return df

def one_hot_encoding(df):
    catcols = ['genre', 'credit_autres', 'cartes_bancaires', 'compte_courant', 'espace_client', 'PEA', 'assurance_auto', 'assurance_habitation', 'credit_immo', 'type', 'methode_contact', 'segment_client', 'branche', 'categorie_age']
    df = pd.get_dummies(df, columns=catcols)
    print('New Number of Features:', df.shape[1])
    return df