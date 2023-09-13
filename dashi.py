import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


# Set Streamlit page configuration
st.set_page_config(
    layout='wide',
    initial_sidebar_state='expanded'
)

## Fetching data 
# Function to fetch client IDs from the API
def fetch_client_ids():
    api_url = 'http://192.168.0.59:5000/predict/'  
    response = requests.get(api_url)

    if response.status_code == 200:
        data = response.json()
        client_ids = list(data.keys())
    else:
        st.error(f"Error fetching client IDs from API. Status code: {response.status_code}")
        st.error(f"Error message from API: {response.text}")
        client_ids = []

    return client_ids

# Function to fetch feature importance from the API
def fetch_feature_importance():
    api_url = 'http://192.168.0.59:5000/feature_importance'
    response = requests.get(api_url)

    if response.status_code == 200:
        feature_importance_data = response.json()
        feature_importance_df = pd.DataFrame({
            'Feature': feature_importance_data['feature_names'],
            'Importance': feature_importance_data['feature_importances']
        })

        # Sort the DataFrame by importance
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        return feature_importance_df
    else:
        st.error(f"Error fetching feature importances from API. Status code: {response.status_code}")
        st.error(f"Error message from API: {response.text}")
        return None


# Title of the dashboard
st.title('Prêt à Dépenser - Credit Scoring Tool')


## Fetching data 

# Fetch client IDs from the API
client_ids = fetch_client_ids()



## Dashboard creation 
# Create a dropdown to select the client ID below the title
selected_client_id = st.selectbox('Select client ID', client_ids)


# Function to update the gauge chart
def update_gauge_chart(sk_id):
    api_url = 'http://192.168.0.59:5000/'
    url_pred = api_url + "predict/" + str(sk_id)
    response = requests.get(url_pred)

    credit_status = '' 

    if response.status_code == 200:
        prediction = int(response.json().get('retour_prediction', 0)) / 100
    else:
        prediction = 0
        st.warning(f"Failed to fetch data for client {sk_id} from the API.")

    # Adjust the value based on the prediction
    if prediction == 0:
        gauge_value = 0.25
        credit_status = 'Favourable'
    elif prediction == 1:
        gauge_value = 0.75
        credit_status = 'Not Favourable'
    else:
        gauge_value = prediction

    # Create a subheader indicating the credit evaluation
    st.subheader('Evaluation')

    # Display the credit status
    st.write(f'Credit Status: {credit_status}')

    # Create the gauge chart with different colors
    gauge_predict = go.Figure(go.Indicator(
        mode="gauge",
        value=gauge_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {
                'range': [0, 1],
                'tickvals': [0, 1],
                'ticktext': ['Favourable', 'Not Favourable']
            },
            'bgcolor': "lightcoral",
            'steps': [
                {'range': [0, 0.5], 'color': 'lightgreen'},
                {'range': [0.5, 1], 'color': 'plum'}
            ],
            'threshold': {
                'line': {'color': "orange", 'width': 4},
                'thickness': 1,
                'value': gauge_value
            }
        }))

    # Display the gauge chart
    st.plotly_chart(gauge_predict)


# Display the gauge chart based on the selected client ID
update_gauge_chart(selected_client_id)

 







