import os

# Chemins des dossiers
CURRENT_DIR = os.getcwd()
PARENT_DIR = os.path.dirname(CURRENT_DIR)
DATA_PATH = os.path.join(PARENT_DIR, "data", "raw", "dataset.csv")
INTERIM_DATA_PATH = os.path.join(PARENT_DIR, "data", "interim", "data.parquet")
PROCESSED_DATA_PATH = os.path.join(PARENT_DIR, "data", "processed", "data.parquet")
MODELS_FOLDER = os.path.join(PARENT_DIR, "models")

RF_IMPORTANT_FEATURES = [
    'age', 'anciennete_mois', 'interet_compte_epargne_total', 'PC1',
    'agios_6mois', 'PC2', 'score_engagement', 'score_risque_financier',
    'PC4'
]
XGB_IMPORTANT_FEATURES = ["assurance_auto_non", "compte_courant_False", "cartes_bancaires_medium", "methode_contact_mail", "categorie_age_65+", "anciennete_mois", "age", "espace_client_non", "segment_client_B3", "segment_client_A3", "segment_client_D2", "segment_client_D3", "segment_client_B4", "segment_client_A2", "PC1", "interet_compte_epargne_total", "type_perso", "compte_titres", "PEA_non"]


# Configuration pandas
PANDAS_DISPLAY_OPTIONS = {
    'display.max_columns': None,
    'display.max_rows': None
}

# Configuration des warnings
IGNORE_WARNINGS = ["ignore", UserWarning]