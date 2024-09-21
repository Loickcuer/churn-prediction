import streamlit as st
import pandas as pd
import plotly.express as px
import io

st.set_page_config(page_title="Prédiction de Churn", page_icon="🔮", layout="wide")

@st.cache_data
def load_data():
    return pd.read_excel("docs/df.xlsx")

@st.cache_data
def load_high_risk_data():
    return pd.read_excel('docs/df_high_risk.xlsx')

df = load_data()
df_high_risk = load_high_risk_data()

st.title("Prédiction du taux d'attrition de la clientèle d'une banque de détail")

# Sidebar pour les informations
with st.sidebar:
    st.header("À propos")
    st.info("Cette application prédit la probabilité de churn d'un client basé sur son ID.")
    
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df_high_risk.to_excel(writer, index=False, sheet_name='Clients à haut risque')
    excel_data = buffer.getvalue()
    
    # Bouton de téléchargement pour le fichier Excel
    st.download_button(
        label="📊 Télécharger les données des clients les plus susceptibles de churner selon les données les plus récentes",
        data=excel_data,
        file_name="clients_haut_risque_churn.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Interface principale
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Sélectionnez l'ID du client")
    client_ids = sorted(df['id_client'].unique())
    selected_id = st.selectbox("ID Client", client_ids)
    client_data = df[df['id_client'] == selected_id]
    
    if not client_data.empty:
        churn_class = client_data['churn_class'].values[0]
        churn_score = client_data['churn_score'].values[0]
        
        st.success(f"Résultat trouvé pour le client {selected_id}")
        
        churn_class_traduction = {
            "Very Low": "très faible",
            "Low": "faible",
            "Average": "moyen",
            "High Risk": "élevé",
            "Very High Risk": "très élevé"
        }
        
        classe_traduite = churn_class_traduction.get(churn_class, churn_class)
        
        # Définition de l'emoji et de la couleur en fonction de la classe
        if churn_class in ["Very Low", "Low"]:
            emoji = "✅"
            message_color = "green"
        elif churn_class == "Average":
            emoji = "⚠️"
            message_color = "orange"
        else:
            emoji = "🚨"
            message_color = "red"
        
        message = f"{emoji} Ce client présente un risque **{classe_traduite}** de churn."
        st.markdown(f"<p style='color:{message_color};'>{message}</p>", unsafe_allow_html=True)
        
        # Affichage des détails du client
        st.subheader("Détails du client")
        
        col_details1, col_details2 = st.columns(2)
        
        with col_details1:
            st.markdown("### Informations personnelles")
            st.write(f"**Âge:** {client_data['age'].values[0]} ans")
            st.write(f"**Genre:** {'Femme' if client_data['genre_F'].values[0] else 'Homme'}")
            st.write(f"**Segment client:** {client_data.filter(regex='^segment_client_').idxmax(axis=1).values[0].split('_')[-1]}")
            st.write(f"**Type de client:** {'Professionnel' if client_data['type_pro'].values[0] else 'Particulier'}")
            st.write(f"**Ancienneté:** {client_data['anciennete_mois'].values[0]} mois")
            
            st.markdown("### Produits bancaires")
            st.write(f"**Compte courant:** {'Oui' if client_data['compte_courant_True'].values[0] else 'Non'}")
            st.write(f"**Compte épargne:** {'Oui' if client_data['compte_epargne'].values[0] else 'Non'}")
            st.write(f"**Compte titres:** {'Oui' if client_data['compte_titres'].values[0] else 'Non'}")
            st.write(f"**PEA:** {'Oui' if client_data['PEA_oui'].values[0] else 'Non' if client_data['PEA_non'].values[0] else 'Inconnu'}")
        
        with col_details2:
            st.markdown("### Engagement et risque")
            st.write(f"**Score d'engagement:** {client_data['score_engagement'].values[0]:.2f}")
            st.write(f"**Score de risque financier:** {client_data['score_risque_financier'].values[0]:.2f}")
            st.write(f"**Agios sur 6 mois:** {client_data['agios_6mois'].values[0]:.2f} €")
            st.write(f"**Intérêts compte épargne:** {client_data['interet_compte_epargne_total'].values[0]:.2f} €")
            
            st.markdown("### Services et produits")
            st.write(f"**Espace client web:** {'Oui' if client_data['espace_client_oui'].values[0] else 'Non' if client_data['espace_client_non'].values[0] else 'Inconnu'}")
            st.write(f"**Assurance vie:** {'Oui' if client_data['assurance_vie'].values[0] else 'Non'}")
            st.write(f"**Assurance auto:** {'Oui' if client_data['assurance_auto_oui'].values[0] else 'Non' if client_data['assurance_auto_non'].values[0] else 'Inconnu'}")
            st.write(f"**Assurance habitation:** {'Oui' if client_data['assurance_habitation_oui'].values[0] else 'Non' if client_data['assurance_habitation_non'].values[0] else 'Inconnu'}")
            st.write(f"**Crédit immobilier:** {'Oui' if client_data['credit_immo_oui'].values[0] else 'Non' if client_data['credit_immo_non'].values[0] else 'Inconnu'}")

with col2:
    st.subheader("Distribution des classes de churn")
    fig = px.pie(df, names='churn_class', title="Répartition des clients", color_discrete_sequence=['#3498db', '#e74c3c'])
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Créé par Loick Cuer")