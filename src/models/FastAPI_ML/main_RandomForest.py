# Commande de lancement de FastAPI: uvicorn main_RandomForest:app --port 8080 --reload
# Voir fichier REDME.txt pour les requêtes curl ou FastAPI Swagger

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import load

from json.decoder import JSONDecodeError

# Chargement du modèle
loaded_model = load('RandomForestClassifier_BestParams.joblib') # Classifier Random Forest entraîné sur 100K exemples d'accidents

# Création d'une nouvelle instance fastAPI
app = FastAPI(

    title="Prédictions gravité des accidents de la route",
    description="API développée avec FastAPI par Christophe Blanchet.",
    version="2.0",
    openapi_tags=[
    {
        'name': 'Main',
        'description': 'Fonctions générales'
    },
        {
        'name': 'Prédictions Gravité Accidents',
        'description': 'Ensemble des fonctions liées à la prédiction ML '
    }]
)


# Définir un objet (une classe) pour réaliser des requêtes
# dot notation (.)
class request_body(BaseModel):
    place   :   float
    catu    :     int
    sexe    :     int
    trajet  :   float
    locp    :   float
    actp    :     int
    etatp   :   float
    cat_age :     int
    secu0   :     int
    secu1   :     int
    secu2   :     int
    secu3   :     int
    secu4   :     int
    secu5   :     int
    secu6   :     int
    secu7   :     int
    secu8   :     int
    secu9   :     int
    catv    :   float
    obs     :   float
    obsm    :   float
    choc    :   float
    manv    :   float
    an      :     int
    mois    :     int
    lum     :     int
    agg     :     int
    int     :     int
    atm     :   float
    col     :   float
    weekday :     int
    hr      :     int
    catr    :   float
    circ    :   float
    vosp    :   float
    prof    :   float
    plan    :   float
    surf    :   float
    infra   :   float
    situ    :   float


##### Point de terminaison pour vérifier que l'API de prédiction est alive #####
@app.get("/api_status", name="Vérification api prédiction alive", tags=['Main'])
def check_api_alive():
    """
    Cette fonction vérifie que l'api de prédiction ML est en ordre de bon fonctionnement.
    """
    return "API: Hello, I'm Ready to predict!"


##### Point de terminaison (API) pour la prédiction sur un accident unique - caractéristiques accidents transmises au format JSON #####
@app.post("/predict", name="Prédiction pour un accident unique", tags=['Prédictions Gravité Accidents']) # local : http://127.0.0.1:8080/predict

# Définition de la fonction de prédiction
def predict(data : request_body):
    """
    Ce point de terminaison permet de lancer une prédiction sur un accident de la route unique:\n

    - Les données de l'accident contenant ses 39 caractéristiques doivent être fournies à l'API sous la forme d'un fichier JSON\n
    - 2 façons de lancer la prédiction pour cet end point:\n
            - Lancement par interface FastAPI Swagger. Voir fichier README.txt pour les 2 requêtes sous format JSON\n
            - Lancement par la fonction: curl -X POST "http://127.0.0.1:8080/predict/"\n
    """
    try:
        # Nouvelle donnée sur laquelle on lance la prédiction
        new_data = [[
            data.place,
            data.catu, 
            data.sexe, 
            data.trajet, 
            data.locp, 
            data.actp, 
            data.etatp, 
            data.cat_age, 
            data.secu0, 
            data.secu1, 
            data.secu2, 
            data.secu3, 
            data.secu4, 
            data.secu5, 
            data.secu6, 
            data.secu7, 
            data.secu8, 
            data.secu9, 
            data.catv, 
            data.obs, 
            data.obsm, 
            data.choc, 
            data.manv, 
            data.an, 
            data.mois, 
            data.lum, 
            data.agg, 
            data.int, 
            data.atm, 
            data.col, 
            data.weekday, 
            data.hr, 
            data.catr, 
            data.circ, 
            data.vosp, 
            data.prof, 
            data.plan, 
            data.surf, 
            data.infra, 
            data.situ
        ]]
        new_data_df = pd.DataFrame(new_data, columns=['place', 'catu', 'sexe', 'trajet', 'locp', 'actp', 'etatp', 'cat_age', 'secu0', 'secu1', 'secu2', 'secu3',
                                                        'secu4', 'secu5', 'secu6', 'secu7', 'secu8', 'secu9', 'catv', 'obs', 'obsm', 'choc', 'manv', 'an', 'mois', 'lum',
                                                        'agg', 'int', 'atm', 'col', 'weekday', 'hr', 'catr', 'circ', 'vosp', 'prof', 'plan', 'surf', 'infra', 'situ'])
        print("\nnew_data_df:\n", new_data_df)

        # Prédiction de probabilité des deux classes
        classe_accident_proba = loaded_model.predict_proba(new_data_df)[0]
        if classe_accident_proba[0] > classe_accident_proba[1]:
            classe_accident = 1
        else:
            classe_accident = 2
        
        return {'classe prédite' : int(classe_accident),
                'Probabilité classe prédite' : float(classe_accident_proba[classe_accident-1])
               }

    except JSONDecodeError:
        # Gérer spécifiquement les erreurs de décodage JSON
        raise HTTPException(status_code=422, detail="Invalid JSON format")
    
    except Exception as e:
        # return {"error": f"An error occurred: {str(e)}"}
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
        # Retourne la classe de l'accident (Classe #1: Blessé & Tués, Classe #2: Indemnes)
    
    
    
##### Point de terminaison (API) pour la prédiction automatisée des nouveaux accidents de la route stockés en base de données #####
@app.post("/predict_auto_batch", name="Prédiction automatique nouveaux accidents en bd", tags=['Prédictions Gravité Accidents']) # local : http://127.0.0.1:8080/predict_auto

# Définition de la fonction de prédiction
def predict_auto_batch(data : request_body):
    """
    Ce point de terminaison permet de lancer automatiquement des prédictions sur les nouveaux accidents de la route stockés en base de données:\n
    """
    # Import des données test accident: data & target
    X_test = pd.read_csv('X_test.zip', nrows=100)
    # print("\nX_test:\n", X_test, "\n")

    y_test = np.array(pd.read_csv('y_test.zip', nrows=100)).ravel()
    # print("y_test:\n",y_test, "\n")


    try:
        # Nouvelle donnée sur laquelle on lance la prédiction
        new_data = [[
            data.place,
            data.catu, 
            data.sexe, 
            data.trajet, 
            data.locp, 
            data.actp, 
            data.etatp, 
            data.cat_age, 
            data.secu0, 
            data.secu1, 
            data.secu2, 
            data.secu3, 
            data.secu4, 
            data.secu5, 
            data.secu6, 
            data.secu7, 
            data.secu8, 
            data.secu9, 
            data.catv, 
            data.obs, 
            data.obsm, 
            data.choc, 
            data.manv, 
            data.an, 
            data.mois, 
            data.lum, 
            data.agg, 
            data.int, 
            data.atm, 
            data.col, 
            data.weekday, 
            data.hr, 
            data.catr, 
            data.circ, 
            data.vosp, 
            data.prof, 
            data.plan, 
            data.surf, 
            data.infra, 
            data.situ
        ]]
        new_data_df = pd.DataFrame(new_data, columns=['place', 'catu', 'sexe', 'trajet', 'locp', 'actp', 'etatp', 'cat_age', 'secu0', 'secu1', 'secu2', 'secu3',
                                                        'secu4', 'secu5', 'secu6', 'secu7', 'secu8', 'secu9', 'catv', 'obs', 'obsm', 'choc', 'manv', 'an', 'mois', 'lum',
                                                        'agg', 'int', 'atm', 'col', 'weekday', 'hr', 'catr', 'circ', 'vosp', 'prof', 'plan', 'surf', 'infra', 'situ'])
        print("\nnew_data_df:\n", new_data_df)

        # Prédiction
        classe_accident = loaded_model.predict(new_data_df)[0]

        # Calcul de la précision (accuracy) de la prédiction
        accuracy = accuracy_score([y_test[0]], [classe_accident])

        # Retourne la classe de l'accident (Classe #1: Blessé & Tués, Classe #2: Indemnes)
        return {'classe prédite' : int(classe_accident),
                'classe réelle'  : int(y_test[0]), # Attention !!!, je prends la première valeur (position 0) de y_test alors que les data sont saisies par le JSON de "FastAPI Sawgger"
                'Précision (accuracy)' : accuracy
            }
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}