#!/bin/bash

# Récupérer les secrets GitHub
username=$1
password=$2

# Générer le fichier .env
echo "STREAMLIT_USERNAME=$username" > .env
echo "STREAMLIT_PASSWORD=$password" >> .env