# Utilisez une image de Python comme base
FROM python:3.9

# Définissez le répertoire de travail dans le conteneur
WORKDIR /app

# Copiez le fichier requirements.txt dans le conteneur
COPY requirements.txt .

# Installez les dépendances Python
RUN pip install -r requirements.txt

# Copiez tous les fichiers de l'application dans le conteneur
COPY . .

# Exposez le port sur lequel votre application écoute
EXPOSE 8501

# Démarrez l'application Streamlit
CMD ["streamlit", "run", "demo_streamlit.py"]