import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
from config import INTERIM_DATA_PATH, PROCESSED_DATA_PATH

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
    df['score_engagement'] = df.apply(calculer_score_engagement, axis=1)
    df = calculer_score_risque_financier(df)
    df = creer_categorie_age(df)
    
    columns_to_pca = [col for col in df.columns if col.startswith('Var_')]
    df = apply_pca(df, columns_to_pca)
    
    return df