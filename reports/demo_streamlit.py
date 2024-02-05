import streamlit as st
import pandas as pd
import os
import time
import requests

from google.auth import default
from google.auth.transport import requests as grequests
from dotenv import load_dotenv

import config

def progress_bar():
        progress_text = "En attente de la réponse de l'API"
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.005)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(1)
        my_bar.empty()

def check_api():
        # Envoie une requête GET à l'API avec les en-têtes appropriés  
    try:
        url = "http://mlopsaccidents.hopto.org/mlapi/api_status"
        headers = {'accept': 'application/json'}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            result = response.json()
            st.write(result)
            st.success("L'API est accessible et fonctionne correctement.", icon="✅")
            
        else:
            return f"Erreur lors de la vérification de l'API. Code de statut : {response.status_code}"

    except requests.exceptions.RequestException as e:
        return f"Erreur lors de la requête GET : {str(e)}"

def make_prediction_Indemnes():
    # Données de prédiction au format JSON
    data = {
        "place": 1,
        "catu": 1,
        "sexe": 2,
        "trajet": 3,
        "locp": 0,
        "actp": 0,
        "etatp": 0,
        "cat_age": 6,
        "secu0": 0,
        "secu1": 1,
        "secu2": 0,
        "secu3": 0,
        "secu4": 0,
        "secu5": 0,
        "secu6": 0,
        "secu7": 0,
        "secu8": 0,
        "secu9": 0,
        "catv": 7,
        "obs": 0,
        "obsm": 0,
        "choc": 4,
        "manv": 2,
        "an": 8,
        "mois": 4,
        "lum": 1,
        "agg": 2,
        "int": 1,
        "atm": 3,
        "col": 2,
        "weekday": 1,
        "hr": 11,
        "catr": 2,
        "circ": 1,
        "vosp": 0,
        "prof": 1,
        "plan": 1,
        "surf": 3,
        "infra": 0,
        "situ": 1
        }

    # URL de l'API
    url = "http://mlopsaccidents.hopto.org/mlapi/predict"

    # En-têtes de la requête
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }

    try:
        # Effectuer la requête POST à l'API
        response = requests.post(url, json=data, headers=headers)

        # Afficher le résultat de la prédiction
        if response.status_code == 200:
            result = response.json()
            st.write("Résultat de la prédiction :")
            st.write(result)
        else:
            st.write(f"Erreur lors de la prédiction. Code de statut : {response.status_code}")

    except requests.exceptions.RequestException as e:
        st.write(f"Erreur lors de la requête POST : {str(e)}")

def make_prediction_Victimes():
    # Données de prédiction au format JSON
    data = {
              "place": 1,
              "catu": 1,
              "sexe": 1,
              "trajet": 0,
              "locp": 0,
              "actp": 0,
              "etatp": 0,
              "cat_age": 3,
              "secu0": 0,
              "secu1": 1,
              "secu2": 0,
              "secu3": 0,
              "secu4": 0,
              "secu5": 0,
              "secu6": 0,
              "secu7": 0,
              "secu8": 0,
              "secu9": 0,
              "catv": 7,
              "obs": 0,
              "obsm": 1,
              "choc": 1,
              "manv": 1,
              "an": 9,
              "mois": 12,
              "lum": 5,
              "agg": 2,
              "int": 2,
              "atm": 1,
              "col": 6,
              "weekday": 6,
              "hr": 17,
              "catr": 4,
              "circ": 2,
              "vosp": 3,
              "prof": 1,
              "plan": 1,
              "surf": 1,
              "infra": 0,
              "situ": 1
        }

    # URL de lAPI
    url = "http://mlopsaccidents.hopto.org/mlapi/predict"

    # En-têtes de la requête
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }

    try:
        # Effectuer la requête POST à l'API
        response = requests.post(url, json=data, headers=headers)

        # Afficher le résultat de la prédiction
        if response.status_code == 200:
            result = response.json()
            st.write("Résultat de la prédiction :")
            st.write(result)
        else:
            st.write(f"Erreur lors de la prédiction. Code de statut : {response.status_code}")

    except requests.exceptions.RequestException as e:
        st.write(f"Erreur lors de la requête POST : {str(e)}")

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Fonction pour l'authentification de l'utilisateur
def authenticate(username, password):
    valid_username = os.getenv("STREAMLIT_USERNAME")
    valid_password = os.getenv("STREAMLIT_PASSWORD")

    if username == valid_username and password == valid_password:
        return True
    else:
        return False

# Interface utilisateur Streamlit
        # Définir le style de la sidebar
sidebar_style = """
    background-color: #3498db; /* Bleu */
    padding: 20px;
"""

# Appliquer le style à la sidebar
st.markdown(
    f"""
    <style>
        .sidebar .sidebar-content {{
            {sidebar_style}
        }}
    </style>
    """,
    unsafe_allow_html=True
)
# Utilisation de la variable de session pour stocker l'état d'authentification
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Afficher le formulaire de connexion si l'utilisateur n'est pas authentifié
if not st.session_state.authenticated:
    username = st.text_input("Nom d'utilisateur")
    password = st.text_input("Mot de passe", type="password")

    if st.button("Se connecter"):
        if authenticate(username, password):
            st.session_state.authenticated = True
            st.success("Connexion réussie !")
        else:
            st.error("Nom d'utilisateur ou mot de passe incorrect.")
else:
    st.success("Vous êtes déjà connecté !")

    # début de l'appli streamlit à proprement parlé
    st.sidebar.title("RouteGuard AI")
    st.sidebar.header("Solution d’assistance de centre d’appel d’urgence routière")
    st.image('./figures/Accident.png')
    st.sidebar.markdown('[Présentation du projet](https://docs.google.com/document/d/1fsapUBaCf9MyIJVW1ClVA07Y4yklb_2CNDxokV93wa0/edit) ')
    # lien gouv si le lien google doc ne fonctionne plus https://www.data.gouv.fr/en/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2022/
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## {config.PROMOTION}")
    st.sidebar.markdown("### Team members:")
    for member in config.TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)

    # Bouton pour vérifier le bon fonctionnement de l'API
    if st.sidebar.button("Vérifier l'API"):
        result_container = st.empty()
        progress_bar()
        check_api()

    # Bouton pour effectuer une prédiction classe indemnes via une requete sur l'API
    if st.sidebar.button("Effectuer une prédiction indemnes"):
        result_container = st.empty()
        indemnes = {
        "place": [1],
        "catu": [1],
        "sexe": [2],
        "trajet": [3],
        "locp": [0],
        "actp": [0],
        "etatp": [0],
        "cat_age": [6],
        "secu0": [0],
        "secu1": [1],
        "secu2": [0],
        "secu3": [0],
        "secu4": [0],
        "secu5": [0],
        "secu6": [0],
        "secu7": [0],
        "secu8": [0],
        "secu9": [0],
        "catv": [7],
        "obs": [0],
        "obsm": [0],
        "choc": [4],
        "manv": [2],
        "an": [8],
        "mois": [4],
        "lum": [1],
        "agg": [2],
        "int": [1],
        "atm": [3],
        "col": [2],
        "weekday": [1],
        "hr": [11],
        "catr": [2],
        "circ": [1],
        "vosp": [0],
        "prof": [1],
        "plan": [1],
        "surf": [3],
        "infra": [0],
        "situ": [1]
    }
        df = pd.DataFrame(indemnes)
        df.rename(index={0: "Jeux de données utilisés"}, inplace=True)
        st.dataframe(df)
        progress_bar()
        make_prediction_Indemnes()

    # Bouton pour effectuer une prédiction classe bléssés et tués via une requete sur l'API
    if st.sidebar.button("Effectuer une prédiction victimes"):
        result_container = st.empty()
        victimes = {
        "place": [1],
        "catu": [1],
        "sexe": [1],
        "trajet": [0],
        "locp": [0],
        "actp": [0],
        "etatp": [0],
        "cat_age": [3],
        "secu0": [0],
        "secu1": [1],
        "secu2": [0],
        "secu3": [0],
        "secu4": [0],
        "secu5": [0],
        "secu6": [0],
        "secu7": [0],
        "secu8": [0],
        "secu9": [0],
        "catv": [7],
        "obs": [0],
        "obsm": [1],
        "choc": [1],
        "manv": [1],
        "an": [9],
        "mois": [12],
        "lum": [5],
        "agg": [2],
        "int": [2],
        "atm": [1],
        "col": [6],
        "weekday": [6],
        "hr": [17],
        "catr": [4],
        "circ": [2],
        "vosp": [3],
        "prof": [1],
        "plan": [1],
        "surf": [1],
        "infra": [0],
        "situ": [1]
    }
        df = pd.DataFrame(victimes)
        df.rename(index={0: "Jeux de données utilisés"}, inplace=True)
        st.dataframe(df)
        progress_bar()
        make_prediction_Victimes()