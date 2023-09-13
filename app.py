from flask import Flask, request, jsonify
import joblib
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import streamlit as st
import numpy as np

mlflow.set_tracking_uri("http://127.0.0.1:5000")  

app = Flask(__name__)

# Load the trained model using MLflow
model_uri = 'runs:/6a773c1e4cda492c80ede5a758f1d636/models/XGBoost_best_model.joblib' 
loaded_model = mlflow.sklearn.load_model(model_uri)

# Load the entire dataframe
df = pd.read_csv('df_sample.csv')

num_client = df.SK_ID_CURR.unique()

@app.route("/")
def index():
    return "App loaded, model and data loaded............" 

@app.route('/predict/')
def predict():
    """
    Returns predictions for all available clients.
    """
    predictions = {}
    for sk_id in num_client:
        predict = loaded_model.predict(df[df['SK_ID_CURR'] == sk_id])[0]
        predict_proba = loaded_model.predict_proba(df[df['SK_ID_CURR'] == sk_id])[0]
        predictions[str(sk_id)] = {
            'prediction': int(predict),
            'predict_proba_0': float(predict_proba[0]),
            'predict_proba_1': float(predict_proba[1])
        }
    
    return jsonify(predictions)

@app.route('/predict/<float:sk_id>')
def predict_get(sk_id):
    """
    Returns
    prediction  0 OK
                1 Default
    """

    if sk_id in num_client:
        predict = loaded_model.predict(df[df['SK_ID_CURR']==sk_id])[0]
        predict_proba = loaded_model.predict_proba(df[df['SK_ID_CURR']==sk_id])[0]
        predict_proba_0 = str(predict_proba[0])
        predict_proba_1 = str(predict_proba[1])
    else:
        predict = predict_proba_0 = predict_proba_1 = "client inconnu"
    return jsonify({ 'retour_prediction' : str(predict), 'predict_proba_0': predict_proba_0,
                     'predict_proba_1': predict_proba_1 })


# Feature importance 




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
