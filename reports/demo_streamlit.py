import streamlit as st
import subprocess
import requests

def start_api():
    # Exécutez la commande uvicorn en arrière-plan
    subprocess.Popen(["uvicorn", "main:api", "--reload"])

def check_api():
    # Envoie une requête GET à l'API avec les en-têtes appropriés
    try:
        url = "http://35.181.233.45:80/api_status"
        headers = {'accept': 'application/json'}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            result = response.json()
            st.write(result)
            return "L'API est accessible et fonctionne correctement."
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
    url = "http://35.181.233.45:80/predict"

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
    url = "http://35.181.233.45:80/predict"

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

# Interface utilisateur Streamlit
st.title("sep23_continu_mops_accidents")
st.image('./figures/Accident.png')
st.markdown('[Présentation du projet](https://docs.google.com/document/d/1fsapUBaCf9MyIJVW1ClVA07Y4yklb_2CNDxokV93wa0/edit) ')
# lien gouv si le lien google doc ne fonctionne plus https://www.data.gouv.fr/en/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2022/

# Bouton pour démarrer l'API
if st.button("Démarrer l'API"):
    st.write("L'API est en cours de démarrage...")
#    result1 = start_api()
    st.write("L'API a été démarrée avec succès!")
 #   st.write(result1)

# Bouton pour vérifier le bon fonctionnement de l'API
if st.button("Vérifier l'API"):
    st.write("Vérification en cours...")
    result = check_api()
    st.write(result)

# Bouton pour effectuer une prédiction classe indemnes via une requete sur l'API
if st.button("Effectuer une prédiction indemnes"):
    st.write("Prédiction en cours...")
    make_prediction_Indemnes()

# Bouton pour effectuer une prédiction classe bléssés et tués via une requete sur l'API
if st.button("Effectuer une prédiction victimes"):
    st.write("Prédiction en cours...")
    make_prediction_Victimes()

