from flask import Flask, request, jsonify
import joblib
import pandas as pd
#import mlflow
#from mlflow.tracking import MlflowClient
#import streamlit as st
import numpy as np
import pickle
import xgboost 
from xgboost import XGBClassifier

#mlflow.set_tracking_uri("http://127.0.0.1:5000")  

app = Flask(__name__)

# Load the trained model using MLflow
#model_uri = 'runs:/faedb810e8e04b7295946d14b1b36949/models/XGBoost_best_model.joblib' 
#loaded_model = mlflow.sklearn.load_model(model_uri)
with open('model.pkl', 'rb')as f:
    loaded_model = joblib.load(f)

# Load the entire dataframe
df = pd.read_csv('df_sample2.csv')

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

####
@app.route('/feature_importance/<float:sk_id>')
def calculate_feature_importance(sk_id):
    """
    Calculate feature importance for a specific client using XGBoost's feature importances.
    """
    if sk_id in num_client:
        # Select the client's data
        client_data = df[df['SK_ID_CURR'] == sk_id]

        # Calculate feature importance using XGBoost's feature importances
        if hasattr(loaded_model, 'feature_importances_'):
            feature_importance = dict(zip(client_data.columns, loaded_model.feature_importances_))

            # Convert values to Python floats
            feature_importance = {k: float(v) for k, v in feature_importance.items()}
        else:
            return jsonify({"error": "Feature importance scores are not available for this model"}), 404

        return jsonify(feature_importance)
    else:
        return jsonify({"error": "Client not found"}), 404


@app.route('/get_client_features/<float:sk_id>')
def get_client_features(sk_id):
    """
    Returns specific client features for display.
    """
    if sk_id in num_client:
        # Select the client's data
        client_data = df[df['SK_ID_CURR'] == sk_id]

        # Extract the desired features and format them
        client_features = {
            'CODE_GENDER': client_data['CODE_GENDER'].values[0],
            'DAYS_BIRTH_years': -client_data['DAYS_BIRTH'].values[0] / 365.0,
            'AMT_INCOME_TOTAL': client_data['AMT_INCOME_TOTAL'].values[0],
            'AMT_CREDIT': client_data['AMT_CREDIT'].values[0],
            'AMT_ANNUITY': client_data['AMT_ANNUITY'].values[0],
            'INCOME_CREDIT_PERC': client_data['INCOME_CREDIT_PERC'].values[0] * 100
        }

        return jsonify(client_features)
    else:
        return jsonify({"error": "Client not found"}), 404


@app.route('/get_dataframe', methods=['GET'])
def get_dataframe():
    """
    Downloads the dataframe for further use
    """
    try:
        # Convert the DataFrame to JSON format
        dataframe_json = df.to_json(orient='records')
        return jsonify(dataframe_json)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get_feature_data/<string:feature_name>')
def get_feature_data(feature_name):
    """
    Returns feature data for the selected feature name.
    """
    if feature_name in df.columns:
        feature_data = df[feature_name].tolist()
        return jsonify({"feature_data": feature_data})
    else:
        return jsonify({"error": "Feature not found"}), 404


@app.route('/client_feature_value/<float:sk_id>/<string:feature_name>')
def fetch_client_feature_value(sk_id, feature_name):
    """
    Fetch and return the client's value for the selected feature.
    """
    if sk_id in num_client:
        # Select the client's data
        client_data = df[df['SK_ID_CURR'] == sk_id]

        # Check if the selected feature exists in the DataFrame
        if feature_name in client_data.columns:
            client_value = client_data[feature_name].values[0]
            return jsonify({"value": client_value})
        else:
            return jsonify({"error": "Feature not found"}), 404
    else:
        return jsonify({"error": "Client not found"}), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
