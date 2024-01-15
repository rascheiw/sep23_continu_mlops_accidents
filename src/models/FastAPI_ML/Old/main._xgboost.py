#pip install fastapi
#pip install uvicorn

# Commande de lancement de FastAPI: uvicorn main_xgboost:app --port 8080 --reload

# Librairies
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

import pandas as pd
import numpy as np
import xgboost as xgb
from joblib import load

# Import des données test accident: data & target
X_test = pd.read_csv('X_test.zip', nrows=100)
print(X_test)

y_test = np.array(pd.read_csv('y_test.zip', nrows=100)).ravel()
y_test = np.where(y_test == 1, 0, y_test)
y_test = np.where(y_test == 2, 1, y_test)
print(y_test)

# Chargement du modèle
loaded_model = load('XGBClassifier_BestParams.joblib') # Classifier XGBoost entraîné sur 100K exemples d'accidents

# Création d'une nouvelle instance fastAPI
app = FastAPI()

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


# Definition du chemin du point de terminaison (API)
@app.post("/predict") # local : http://127.0.0.1:8000/predict

# Définition de la fonction de prédiction
def predict(data : request_body):
    # Nouvelles données sur lesquelles on fait la prédiction
    new_data = [[
        data.place,
        data.catu   , 
        data.sexe   , 
        data.trajet , 
        data.locp   , 
        data.actp   , 
        data.etatp  , 
        data.cat_age, 
        data.secu0  , 
        data.secu1  , 
        data.secu2  , 
        data.secu3  , 
        data.secu4  , 
        data.secu5  , 
        data.secu6  , 
        data.secu7  , 
        data.secu8  , 
        data.secu9  , 
        data.catv   , 
        data.obs    , 
        data.obsm   , 
        data.choc   , 
        data.manv   , 
        data.an     , 
        data.mois   , 
        data.lum    , 
        data.agg    , 
        data.int    , 
        data.atm    , 
        data.col    , 
        data.weekday, 
        data.hr     , 
        data.catr   , 
        data.circ   , 
        data.vosp   , 
        data.prof   , 
        data.plan   , 
        data.surf   , 
        data.infra  , 
        data.situ
    ]]

    # Prédiction
    class_accident = loaded_model.predict(new_data)[0]

    # Je retourne le nom de l'espèce iris
    return {'class' : class_accident}