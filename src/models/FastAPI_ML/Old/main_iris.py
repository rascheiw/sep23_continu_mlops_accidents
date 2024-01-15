# Commande de lancement de FastAPI: uvicorn main_iris:app --port 8080 --reload
# Commande d'exécution prédiciton avec choix d'Iris correspondant à la ligne du dataset: curl -X POST "http://127.0.0.1:8080/predict/{choix_iris:int}"

# Librairies
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from sklearn.datasets import load_iris
import xgboost as xgb
from joblib import load

iris = load_iris()
print('\nData Iris:\n', iris.data[0:5], '\n')

# Chargement du modèle
loaded_model = load('logreg.joblib')

# Création d'une nouvelle instance fastAPI
app = FastAPI()

# Définir un objet (une classe) pour réaliser des requêtes
# dot notation (.)
# class request_body(BaseModel):
#     sepal_length : float
#     sepal_width : float
#     petal_length : float
#     petal_width : float

# Definition du chemin du point de terminaison (API)
@app.post("/predict/{choix_iris:int}") # local : http://127.0.0.1:8000/predict

# Définition de la fonction de prédiction
def predict(choix_iris):
    # Nouvelles données sur lesquelles on fait la prédiction
    new_data = [[
        iris.data[choix_iris,0],
        iris.data[choix_iris,1],
        iris.data[choix_iris,2],
        iris.data[choix_iris,3]
    ]]

    # Prédiction
    class_idx = loaded_model.predict(new_data)[0]
    print("\nLigne:", choix_iris, "\nnew_data:", new_data, '\n')

    # Afficher le résultat avant de le retourner
    prediction_result = {'class': iris.target_names[class_idx]}
    print("Résultat de la prédiction:", prediction_result)

    # Je retourne le nom de l'espèce iris
    return prediction_result, new_data

# data_to_predict = {
#     "sepal_length": iris.data[0,0],
#     "sepal_width": iris.data[0,1],
#     "petal_length": iris.data[0,2],
#     "petal_width": iris.data[0,3]
# }


# predict(request_body(**data_to_predict))
