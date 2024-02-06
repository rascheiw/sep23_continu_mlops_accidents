# Imge python
FROM python:3.9

LABEL Raphaël SCHEIWILLER <rascheiw@gmail.com>

# Répertoire de travail du conteneur
WORKDIR /app

# Copie du fichier de requirements
COPY requirements.txt .

# Installation des dépendances Python
RUN pip install -r requirements.txt

# Copie des fichiers de l'application dans le conteneur
COPY config.py members.py demo_streamlit.py .env .
COPY figures/Accident.png ./figures/

# Exposition du port 8501
EXPOSE 8501

# Démarrage de l'application Streamlit
CMD ["streamlit", "run", "demo_streamlit.py"]