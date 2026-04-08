# credit-risk-mlops
Credit Risk Prediction - MLOps Project

Objectif:
Prédire le risque de défaut de paiement d’un client à partir de ses caractéristiques financières.

Dataset
Variables financières (income, FICO, dti…)

Target : not.fully.paid

Modèles testés:
-Logistic Regression 
-Decision Tree
-Random Forest

Tracking avec MLflow
Un modèle = un experiment
Plusieurs runs = itérations

Métriques :
precision
recall
f1-score

Modèle sélectionné: Logistic Regression avec class_weight="balanced"

Pourquoi ?
Meilleur recall (détection des défauts)
Modèle interprétable
Bonne généralisation

Résultats:
Model	                Recall	        Precision
Logistic Regression	     0.67	         0.26
Decision Tree	         0.46	         0.25
Random Forest	         0.02	         0.42

Application Streamlit
Interface permettant :
- saisie des données client
- prédiction du risque
- affichage de la probabilité


Lancer le projet:
pip install -r requirements.txt
streamlit run app/streamlit_app.py

## Deployment

The application is deployed on AWS ECS and is publicly accessible at:

http://13.38.66.245:8501/