# Librairies de base
import pandas as pd
import numpy as np
import streamlit as st
import joblib

# Charger le modèle SVR
svr_model = joblib.load("SVR_pipeline.pkl")

# Charger les données pour les options de saisie
df = pd.read_csv("salaries_cleaned.csv")

# Configuration de l'application Streamlit
st.set_page_config(page_title='Prédictions de Salaire avec SVR', layout='wide', initial_sidebar_state='expanded')

# Page de prédiction avec SVR
st.title("🔍 Prédictions de Salaire avec SVR")

# Section pour la saisie des données utilisateur
st.markdown("<h3 style='font-size:20px;'>Saisir les Caractéristiques de l'Usager :</h3>", unsafe_allow_html=True)
with st.form("prediction_form"):
    work_year = st.number_input("**Année de Travail (2020-2024)**", min_value=2020, max_value=2024, step=1)
    salary_currency = st.selectbox("**Devise du Salaire (USD, EUR, GBP, etc.)**", df['salary_currency'].unique())
    experience_level = st.selectbox("**Niveau d'Expérience (Débutant, Intermédiaire, Cadre, Senior)**",
                                    ['Débutant', 'Intermédiaire', 'Cadre', 'Senior'])
    job_title = st.selectbox("**Titre du Poste**", df['job_title'].unique())
    employee_residence = st.selectbox("**Résidence de l'Usager**", df['employee_residence'].unique())
    remote_ratio = st.number_input("**Ratio de Télétravail (0-100)**", min_value=0, max_value=100, step=1)
    company_location = st.selectbox("**Localisation de l'Entreprise**", df['company_location'].unique())
    company_size = st.selectbox("**Taille de l'Entreprise (S, M, L)**", df['company_size'].unique())
    employment_type = st.selectbox("**Type d'emploi**", df['employment_type'].unique())

    submitted = st.form_submit_button("**Prédire**")

if submitted:
    input_data = pd.DataFrame({
        'work_year': [work_year],
        'salary_currency': [salary_currency],
        'experience_level': [experience_level],
        'job_title': [job_title],
        'employee_residence': [employee_residence],
        'remote_ratio': [remote_ratio],
        'company_location': [company_location],
        'company_size': [company_size],
        'employment_type': [employment_type]
    })
    st.write("### Données Saisies :")
    st.write(input_data)

    # Prédiction avec le modèle SVR
    prediction = svr_model.predict(input_data)
    prediction_usd = np.expm1(prediction)  # Convert back from log scale
    prediction_eur = prediction_usd * 0.85  # Convert to EUR using exchange rate 1 USD = 0.85 EUR

    st.write(f"### Salaire Prédit en Dollars : $ {prediction_usd[0]:,.2f}")
    st.write(f"### Salaire Prédit en Euros : € {prediction_eur[0]:,.2f}")

# Ajout de style
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Ajout d'un pied de page
st.markdown("""
    <style>
    footer {
        visibility: hidden;
    }
    footer:after {
        content:'Développé par Votre Nom';
        visibility: visible;
        display: block;
        position: relative;
        padding: 5px;
        top: 2px.
    }
    </style>
    """, unsafe_allow_html=True)
