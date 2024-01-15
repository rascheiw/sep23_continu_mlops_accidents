pip install fastapi
pip install uvicorn

pip install pydantic

pip install scikit-learn
pip install pandas
pip install numpy
pip install xgboost
pip install joblib


##### Prédiction par interface FastAPI Swagger: #####

  ### Classe réelle = 2 (Indemnes) ###
{
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

  ### Classe réelle = 1 (Blessés & Tués) ###
{
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


##### Prédiction par curl: #####

### Démarrer le service ###
/home/ubuntu/venv/bin/activate
uvicorn main_RandomForest:app --port 8080 --reload


  ### Test API is ready to predict ###

curl -X 'GET' \
  'http://127.0.0.1:8080/api_status'


  ### Classe réelle = 2 (Indemnes) ###

curl -X POST -i \
"http://127.0.0.1:8080/predict" \
     -H 'Content-Type: application/json' \
     -d '{
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
         }'

  ### Classe réelle = 1 (Blessés & Tués) ###

curl -X POST -i \
"http://127.0.0.1:8080/predict" \
     -H 'Content-Type: application/json' \
     -d '{
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
		 }'