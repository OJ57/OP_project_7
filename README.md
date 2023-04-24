# OpenClassrooms Parcours Data Scientist - Projet 7
# Implémentez un modèle de scoring

Bienvenue dans le dépôt GitHub du projet 7 du parcours Data Scientist d'OpenClassrooms, "*Implémentez un modèle de scoring*".

Ce projet vise à développer un modèle de scoring pour prédire la probabilité de défaut de paiement d'un client.

## Présentation du projet

La société financière "Prêt à dépenser" propose des crédits à la consommation pour des personnes ayant
peu ou pas du tout d’historique de prêt. L’entreprise souhaite mettre en oeuvre un outil de scoring crédit
pour calculer la probabilité qu’un client rembourse son crédit et classifier la demande en crédit accordé ou
refusé. L’objectif est de développer un algorithme de classification en s’appuyant sur les données de la compétition Kaggle ["Home Credit Default Risk"](https://www.kaggle.com/competitions/home-credit-default-risk). 


## Structure du projet

Le dossier __API__ contient les données nécessaires à l'API de prédiction, notamment le script de l'API 'API_script.py'
et celui des tests unitaires 'test_uni_API_script.py'.

Le dossier __Dashboard__ contient les données nécessaires au Dashboard, notamment le script du Dashboard 'dashboard.py'.

Le dossier __Note_methodologique__ contient la note méthodologique décrivant la démarche d’élaboration du modèle.

__Evidently_report_full.rar__ : Rapport html sur le data drift effectué avec la librairie Evidently.

Les quatres __notebooks Jupyter__ décrivent :

1) L'analyse exploratoire et nettoyage du fichier principal
2) L'analyse exploratoire et nettoyage des fichiers secondaires, scripts et fusion des jeux de données
3) Le preprocessing du jeu de données final : imputation des valeurs manquantes et features selection
4) La modelisation : sélection du modèle, optimisation et interprétabilité

L'API est disponible à l'adresse suivante : https://fastapi-project7.herokuapp.com/
Le Dashboard : https://dashboard-p7.herokuapp.com/
