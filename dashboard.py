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


# Title of the dashboard
st.title('Prêt à Dépenser - Credit Scoring Tool')


## Fetching and displaying Client ID 
# Function to fetch client IDs from the API
def fetch_client_ids():
    api_url = 'http://opc7-eh-97fdd39e1bdc.herokuapp.com/predict/'  
    response = requests.get(api_url)

    if response.status_code == 200:
        data = response.json()
        client_ids = list(data.keys())
    else:
        st.error(f"Error fetching client IDs from API. Status code: {response.status_code}")
        st.error(f"Error message from API: {response.text}")
        client_ids = []

    return client_ids

# Fetch client IDs from the API
client_ids = fetch_client_ids()
 
# Create a dropdown to select the client ID below the title
selected_client_id = st.selectbox('Select client ID', client_ids)


## Organisation of the dashboard 
# Create a two-column layout
col1, col2 = st.columns(2)


## Fetching and displaying predictions in a gauge chart for credit approval
# Function to update the gauge chart
def update_gauge_chart(sk_id):
    api_url = 'http://opc7-eh-97fdd39e1bdc.herokuapp.com/'
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
        credit_status = 'Favorable'
    elif prediction == 1:
        gauge_value = 0.75
        credit_status = 'Not Favorable'
    else:
        gauge_value = prediction

    # Create a subheader indicating the credit evaluation
    st.subheader('Evaluation')

    # Display the credit status 
    st.markdown(f'**Credit Status:** <span style="font-size:30px">{credit_status}</span>', unsafe_allow_html=True)

    # Create the gauge chart with different colors
    gauge_predict = go.Figure(go.Indicator(
        mode="gauge",
        value=gauge_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {
                'range': [0, 1],
                'tickvals': [0, 1],
                'ticktext': ['Fav.', 'Not Fav.']
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
with col1: 
    update_gauge_chart(selected_client_id)


## Global feature importance 
# Display img
with col1: 
    st.image('shap_values.png', caption='Global Feature Importance', use_column_width=True) 


## Fetch and display client info
# Function to fetch client info
def fetch_and_display_client_features(sk_id):
    # Make an API request to get client features
    response = requests.get(f'http://opc7-eh-97fdd39e1bdc.herokuapp.com/get_client_features/{sk_id}')

    if response.status_code == 200:
        client_features = response.json()

        # Display the retrieved client info
        st.subheader('Client Information')
        st.write(f'Code Gender: {client_features["CODE_GENDER"]}')
        st.write(f'Client Age (years): {client_features["DAYS_BIRTH_years"]:.2f}')
        st.write(f'Client Income ($): {client_features["AMT_INCOME_TOTAL"]}')
        st.write(f'Credit Amount of the Loan ($): {client_features["AMT_CREDIT"]}')
        st.write(f'Loan Annuity ($): {client_features["AMT_ANNUITY"]}')
        st.write(f'Annuity as Percentage of Total Income (%): {client_features["INCOME_CREDIT_PERC"]}')
    else:
        st.warning(f"Failed to fetch client features. Status code: {response.status_code}")

# Display client features for the selected client
with col2: 
    if selected_client_id:
        fetch_and_display_client_features(selected_client_id)




## Global Feature distribution and client positioning 
# Function to fetch the entire DataFrame from the API
def fetch_dataframe_from_api():
    api_url = 'http://opc7-eh-97fdd39e1bdc.herokuapp.com/get_dataframe'  
    response = requests.get(api_url)

    if response.status_code == 200:
        json_data = response.json()
        df = pd.read_json(json_data)  
        return df
    else:
        st.error(f"Error fetching DataFrame from API. Status code: {response.status_code}")
        return None


# Load the DataFrame
df = fetch_dataframe_from_api()

# Function to fetch feature distribution
def fetch_client_feature_value(client_id, feature_name):
    api_url = f'http://opc7-eh-97fdd39e1bdc.herokuapp.com/client_feature_value/{client_id}/{feature_name}'
    response = requests.get(api_url)

    if response.status_code == 200:
        data = response.json()
        client_value = data.get('value')
        return client_value
    else:
        st.warning(f"Failed to fetch client feature value. Status code: {response.status_code}")
        return None


# Create a 2-column layout for Feature 1 and Feature 2
col1, col2 = st.columns(2)

# Feature 1
with col1:
    st.subheader('Global Distribution & Client Positioning - Feature 1')
    if df is not None:
        # Create a dropdown to select Feature 1
        selected_feature = st.selectbox('Select Feature 1', df.columns, key="unique_key_for_selectbox_1")
    else:
        st.warning("DataFrame not loaded. Please check the API connection.")
    
    # Display the distribution of Feature 1 and the client positioning
    if selected_feature:
        # Plot the feature distribution
        fig = px.histogram(df, x=selected_feature, title=f'Distribution of {selected_feature}')
        
        # Fetch and display the client's value for Feature 1
        client_value = fetch_client_feature_value(selected_client_id, selected_feature)
        if client_value is not None:
            # Add a marker or text annotation for the client's value
            fig.add_trace(go.Scatter(x=[client_value], y=[0], mode='markers+text', text=[f'Client Value: {client_value}'], textposition='top right', marker=dict(size=10, color='red')))
        
        st.plotly_chart(fig)

# Feature 2
with col2:
    st.subheader('Global Distribution & Client Positioning - Feature 2')
    
    if df is not None:
        # Create a dropdown to select Feature 2
        selected_feature_2 = st.selectbox('Select Feature 2', df.columns, key="unique_key_for_selectbox_2")
    else:
        st.warning("DataFrame not loaded. Please check the API connection.")
    
    # Display the distribution of Feature 2 and the client positioning
    if selected_feature_2:
        # Plot the feature distribution
        fig_2 = px.histogram(df, x=selected_feature_2, title=f'Distribution of {selected_feature_2}')
        
        # Fetch and display the client's value for Feature 2
        client_value_2 = fetch_client_feature_value(selected_client_id, selected_feature_2)
        if client_value_2 is not None:
            # Add a marker or text annotation for the client's value
            fig_2.add_trace(go.Scatter(x=[client_value_2], y=[0], mode='markers+text', text=[f'Client Value: {client_value_2}'], textposition='top right', marker=dict(size=10, color='red')))
        
        st.plotly_chart(fig_2)


# Create a scatter plot for the bivariate analysis
scatter_fig = px.scatter(df, x=selected_feature, y=selected_feature_2, title=f'Bivariate Analysis: {selected_feature} vs. {selected_feature_2}')

# Fetch and display the client's values for the selected features
client_value_1 = fetch_client_feature_value(selected_client_id, selected_feature)
client_value_2 = fetch_client_feature_value(selected_client_id, selected_feature_2)

# Add a marker or text annotation for the client's values
if client_value_1 is not None and client_value_2 is not None:
    scatter_fig.add_trace(go.Scatter(x=[client_value_1], y=[client_value_2], mode='markers', 
                                     marker=dict(size=10, color='red')))
elif client_value_1 is not None:
    scatter_fig.add_trace(go.Scatter(x=[client_value_1], y=[client_value_2], mode='markers', 
                                     marker=dict(size=10, color='red')))
elif client_value_2 is not None:
    scatter_fig.add_trace(go.Scatter(x=[client_value_1], y=[client_value_2], mode='markers', 
                                     marker=dict(size=10, color='red')))

st.plotly_chart(scatter_fig)