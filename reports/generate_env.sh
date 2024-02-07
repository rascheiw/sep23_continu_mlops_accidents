#!/bin/bash

# Récupérer les arguments
username_env=$1
password_env=$2

# Vérifier si les variables sont définies
if [ -z "$username_env" ] || [ -z "$password_env" ]; then
  echo "Erreur: les variables ne sont pas définies."
  exit 1
fi

# Générer le fichier .env
echo "STREAMLIT_USERNAME=$username_env" > .env
echo "STREAMLIT_PASSWORD=$password_env" >> .env