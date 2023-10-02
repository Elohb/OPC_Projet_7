# Projet_7
Projet 7 Openclassrooms formation Data Scientist 

Mission : 
L’entreprise 'Prêt à dépenser' souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).

Objectifs du projet : 
1. Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.
2. Construire un dashboard interactif à destination des gestionnaires de la relation client permettant d'interpréter les prédictions faites par le modèle, et d’améliorer la connaissance client des chargés de relation client.
3. Mettre en production le modèle de scoring de prédiction à l’aide d’une API, ainsi que le dashboard interactif qui appelle l’API pour les prédictions.

Fichiers : 
- XGBoost_model : Données du modèle XGBoost Classifier final
- backend : API et dashboard tests + Notebook prétraitement data, données csv, pkl et png utilisées pour leur développement 
- frontend : API et dashboard complets + Notebook prétraitement data, données csv, pkl et png utilisées pour leur fonctionnement
- main : API et dashboard utilisés pour le déploiement (sans feature importance locales) + Notebook prétraitement data, données csv, pkl et png utilisées pour leur déploiement sur Heroku et streamlit 
