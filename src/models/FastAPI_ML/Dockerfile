# Utilisation d'Alpine Linux comme base
FROM python:3.11-alpine

LABEL Christophe Blanchet <blanchcn@gmail.com>

# Définition du répertoire de travail
WORKDIR /ml_scripts/

# Copie des fichiers requis dans le conteneur
COPY main_RandomForest.py RandomForestClassifier_BestParams.joblib requirements.txt /ml_scripts/

# Mise à jour des paquets et installation de Python 3 et pip3
RUN apk update && apk add --no-cache build-base python3 py3-pip

# Création et activation de l'environnement virtuel
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"
RUN source /venv/bin/activate

# Installation des dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposition du port 8080
EXPOSE 8080

# Commande pour lancer FastAPI avec le fichier main_RandomForest.py
CMD ["uvicorn", "main_RandomForest:app", "--host", "0.0.0.0", "--port", "8080"]


