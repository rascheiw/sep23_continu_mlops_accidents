# Imge python
FROM python:3.9

LABEL Raphaël SCHEIWILLER <rascheiw@gmail.com>

# Répertoire de travail du conteneur
WORKDIR /app

# Copie du fichier de requirements
COPY requirements.txt .

# Installation des dépendances Python
RUN pip install -r requirements.txt

# Copie du script de génération de .env
COPY generate_env.sh .
RUN chmod +x generate_env.sh

# Génération du fichier.env
RUN ./generate_env.sh $username $password

# Copie des fichiers de l'application dans le conteneur
COPY config.py members.py demo_streamlit.py ./
COPY figures/Accident.png ./figures/

# Exposition du port 8501
EXPOSE 8501

# Démarrage de l'application Streamlit
CMD ["streamlit", "run", "demo_streamlit.py"]