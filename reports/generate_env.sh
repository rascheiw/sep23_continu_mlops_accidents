#!/bin/bash

# Récupérer les arguments
username_env=$1
password_env=$2

# Générer le fichier .env
echo "STREAMLIT_USERNAME=$username_env" > .env
echo "STREAMLIT_PASSWORD=$password_env" >> .env