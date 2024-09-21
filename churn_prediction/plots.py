import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

def set_style(): # Définit le style des graphiques.
    plt.style.use('seaborn')
    plt.rcParams['figure.figsize'] = (10, 6)

def plot_correlation_heatmap(df, columns=None): # Trace la matrice de corrélation pour les colonnes spécifiées.
    set_style()
    if columns is None:
        columns = df.columns
    corr = df[columns].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0, center=0)
    plt.title('Matrice de corrélation')
    plt.show()

def plot_feature_importance(importance, names, model_type): # Trace l'importance des features pour un modèle donné.
    set_style()
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    plt.figure(figsize=(10, 8))
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    plt.title(f'FEATURE IMPORTANCE {model_type}')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.show()

def plot_roc_curve(y_test, y_pred_proba, model_name): # Trace la courbe ROC pour un modèle donné.
    set_style()
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.show()

def plot_feature_distribution(df, feature, target='churn'): # Trace la distribution d'une feature numérique en fonction de la variable cible.
    set_style()
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=feature, hue=target, kde=True, multiple="stack")
    plt.title(f'Distribution de {feature} par {target}')
    plt.show()

def plot_categorical_feature(df, feature, target='churn'): # Trace la distribution d'une feature catégorielle en fonction de la variable cible.
    set_style()
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=feature, hue=target)
    plt.title(f'Distribution de {feature} par {target}')
    plt.xticks(rotation=45)
    plt.show()

if __name__ == "__main__":
    # Exemple d'utilisation
    df = pd.read_parquet("../data/interim/data.parquet")
    plot_correlation_heatmap(df, ['age', 'anciennete_mois', 'score_engagement', 'score_risque_financier'])
    plot_feature_distribution(df, 'age')
    plot_categorical_feature(df, 'categorie_age')