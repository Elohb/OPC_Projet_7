from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import pickle
import xgboost 
from xgboost import XGBClassifier
#import shap 

app = Flask(__name__)

#Loading model information
with open('model.pkl', 'rb')as f:
    loaded_model = joblib.load(f)


# Load the entire dataframe
df = pd.read_csv('df_sample2.csv')


# Extract all feature names 
feature_names = list(df.columns)

# Extract client ID
num_client = df.SK_ID_CURR.unique()


@app.route("/")
def index():
    """
    Checking the API is working
    """
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
    Returns prediction and prediction probabilities for a given client.
    """
    if str(sk_id) in shap_values_dict:
        # Get the XGBoost model's prediction and prediction probabilities
        client_data = df[df['SK_ID_CURR'] == sk_id][feature_names]
        predict = loaded_pipeline.predict(client_data)[0]
        predict_proba = loaded_pipeline.predict_proba(client_data)[0]
        predict_proba_0 = str(predict_proba[0])
        predict_proba_1 = str(predict_proba[1])
    else:
        predict = predict_proba_0 = predict_proba_1 = "client inconnu"

    return jsonify({
        'retour_prediction': str(predict),
        'predict_proba_0': predict_proba_0,
        'predict_proba_1': predict_proba_1
    })



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
