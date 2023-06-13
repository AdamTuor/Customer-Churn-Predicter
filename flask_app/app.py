from dash import Dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import ETL
from joblib import load
import shap
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from IPython.display import HTML
import matplotlib.pyplot as plt
import numpy as np

# Initialize Flask server
#server = Flask(__name__)

# Initialize Dash app
app = Dash(__name__)
server = app.server

# Define Dash layout
app.layout = html.Div(
    style={
        "background-image": 'url("https://www.appier.com/hubfs/Imported_Blog_Media/GettyImages-1030850238-01.jpg")',
        "background-size": "cover",
        "background-position": "center",
        "background-attachment": "fixed",
    },
    children=[
        html.H1(
            "Telecom Details",
            style={
                "color": "Orange",
                "font-size": "50px",
                "text-align": "left",
                "margin-top": "10px",
                "padding": "30px 10px",
                "margin-bottom": "10px",
                "font-weight":"bold",
                "text-decoration": "underline",
            },
        ),
    html.H2('Choose your gender:', style={"color":"orange","margin-left":"10px"}),
    dcc.Dropdown(id='gender-dropdown',options=[{'label':'Male','value':'Male'},{'label':'Female','value':'Female'}], placeholder='Select Gender',style={"width": "25%","margin-left":"10px"}),
    html.H2('Are you a senior citizen:', style={"color":"orange","margin-left":"10px"}),
    dcc.Dropdown(id='senior-dropdown',options=[{'label':'Yes','value':1},{'label':'No','value':0}], placeholder='Select Status',style={"width": "25%","margin-left":"10px"}),
    html.H2('Do you have a partner:', style={"color":"orange","margin-left":"10px"}),
    dcc.Dropdown(id='partner-dropdown',options=[{'label':'Yes','value':'Yes'},{'label':'No','value':'No'}], placeholder='Select Partner Status',style={"width": "25%","margin-left":"10px"}),
    html.H2('Do you have dependents:', style={"color":"orange","margin-left":"10px"}),
    dcc.Dropdown(id='dependents-dropdown',options=[{'label':'Yes','value':'Yes'},{'label':'No','value':'No'}], placeholder='Select Dependents Status',style={"width": "25%","margin-left":"10px"}),
    html.H2('Length of tenure(months):', style={"color":"orange","margin-left":"10px","width":"15%"}),
    dcc.Input(id='tenure',type='text',placeholder='Enture tenure(months)',style={"margin-left":"10px"}),
    html.H2('Do you have phone service:', style={"color":"orange","margin-left":"10px"}),
    dcc.Dropdown(id='phone-service-dropdown', options=[{'label': 'Yes', 'value': 'Yes'}, {'label': 'No', 'value': 'No'}], placeholder='Select Phone Service',style={"width": "25%","margin-left":"10px"}),
    html.H2('Do you have multiple lines:', style={"color":"orange","margin-left":"10px"}),
    dcc.Dropdown(id='multiple-lines-dropdown', options=[{'label': 'No phone service', 'value': 'No phone service'},{'label': 'Yes', 'value': 'Yes'}, {'label': 'No', 'value': 'No'}], placeholder='Select Multiple Lines',style={"width": "25%","margin-left":"10px"}),
    html.H2('Do you have internet service:', style={"color":"orange","margin-left":"10px"}),
    dcc.Dropdown(id='internet-service-dropdown', options=[{'label': 'DSL', 'value': 'DSL'},{'label': 'Fiber optic', 'value': 'Fiber optic'}, {'label': 'No', 'value': 'No'}], placeholder='Select Internet Service',style={"width": "25%","margin-left":"10px"}),
    html.H2('The following questions are about internet services, if you selected No above please select No internet service for accuracte results.',style={"color":"white","margin-left":"10px"}),
    html.H2('Online security:', style={"color":"orange","margin-left":"10px"}),
    dcc.Dropdown(id='online-security-dropdown', options=[{'label': 'Yes', 'value': 'Yes'},{'label': 'No', 'value': 'No'}, {'label': 'No internet service', 'value': 'No internet service'}], placeholder='Select Online Security',style={"width": "25%","margin-left":"10px"}),
    html.H2('Online backup:', style={"color":"orange","margin-left":"10px"}),
    dcc.Dropdown(id='online-backup-dropdown', options=[{'label': 'Yes', 'value': 'Yes'},{'label': 'No', 'value': 'No'}, {'label': 'No internet service', 'value': 'No internet service'}], placeholder='Select Online Backup',style={"width": "25%","margin-left":"10px"}),
    html.H2('Device protection:', style={"color":"orange","margin-left":"10px"}),
    dcc.Dropdown(id='device-protection-dropdown', options=[{'label': 'Yes', 'value': 'Yes'},{'label': 'No', 'value': 'No'}, {'label': 'No internet service', 'value': 'No internet service'}], placeholder='Select Device Protection',style={"width": "25%","margin-left":"10px"}),
    html.H2('Tech support:', style={"color":"orange","margin-left":"10px"}),
    dcc.Dropdown(id='tech-support-dropdown', options=[{'label': 'Yes', 'value': 'Yes'},{'label': 'No', 'value': 'No'}, {'label': 'No internet service', 'value': 'No internet service'}], placeholder='Select Tech Support',style={"width": "25%","margin-left":"10px"}),
    html.H2('Streaming TV:', style={"color":"orange","margin-left":"10px"}),
    dcc.Dropdown(id='streaming-tv-dropdown', options=[{'label': 'Yes', 'value': 'Yes'},{'label': 'No', 'value': 'No'}, {'label': 'No internet service', 'value': 'No internet service'}], placeholder='Select Streaming TV',style={"width": "25%","margin-left":"10px"}),
    html.H2('Streaming movies:', style={"color":"orange","margin-left":"10px"}),
    dcc.Dropdown(id='streaming-movies-dropdown', options=[{'label': 'Yes', 'value': 'Yes'},{'label': 'No', 'value': 'No'}, {'label': 'No internet service', 'value': 'No internet service'}], placeholder='Select Streaming Movies',style={"width": "25%","margin-left":"10px"}),
    html.H2('Contract type:', style={"color":"orange","margin-left":"10px"}),
    dcc.Dropdown(id='contract-dropdown', options=[{'label': 'Month-to-month', 'value': 'Month-to-month'},{'label': 'One year', 'value': 'One year'}, {'label': 'Two year', 'value': 'Two year'}], placeholder='Select Contract Type',style={"width": "25%","margin-left":"10px"}),
    html.H2('Paperless billing:', style={"color":"orange","margin-left":"10px"}),
    dcc.Dropdown(id='paperless-billing-dropdown', options=[{'label': 'Yes', 'value': 'Yes'}, {'label': 'No', 'value': 'No'}], placeholder='Select Paperless Billing',style={"width": "25%","margin-left":"10px"}),
    html.H2('Payment method:', style={"color":"orange","margin-left":"10px"}),
    dcc.Dropdown(id='payment-method-dropdown', options=[{'label': 'Electronic check', 'value': 'Electronic check'},{'label': 'Mailed check', 'value': 'Mailed check'}, {'label': 'Bank transfer (automatic)', 'value': 'Bank transfer (automatic)'},{'label': 'Credit card (automatic)', 'value': 'Credit card (automatic)'}], placeholder='Select Payment Method',style={"width": "25%","margin-left":"10px"}),
    html.H2('Monthly charges:', style={"color":"orange","margin-left":"10px"}),
    dcc.Input(id='monthly-charges',type='text',placeholder='Enture Monthly Charges',style={"width": "15%","margin-left":"10px"}),
    html.H2('Total charges:', style={"color":"orange","margin-left":"10px"}),
    dcc.Input(id='total-charges',type='text',placeholder='Enture Total Charges',style={"width": "15%","margin-left":"10px"}),
    html.Button('Predict', id='predict-button', n_clicks=0,style={"background-color": "#4CAF50",
                        "border": "none",
                        "color": "Black",
                        "padding": "15px 32px",
                        "text-align": "center",
                        "text-decoration": "none",
                        "display": "inline-block",
                        "font-size": "20px",
                        "margin": "4px 2px",
                        "cursor": "pointer",
                        "width": "100%",
                        "font-weight":"bold",
}),
    
    html.Div(id='shap-plot')

])

# Define Dash callback
@app.callback(
    Output('shap-plot', 'children'),
    Input('predict-button', 'n_clicks'),
    [
        State('gender-dropdown', 'value'),
        State('senior-dropdown', 'value'),
        State('partner-dropdown', 'value'),
        State('dependents-dropdown', 'value'),
        State('tenure', 'value'),
        State('phone-service-dropdown', 'value'),
        State('multiple-lines-dropdown', 'value'),
        State('internet-service-dropdown', 'value'),
        State('online-security-dropdown', 'value'),
        State('online-backup-dropdown', 'value'),
        State('device-protection-dropdown', 'value'),
        State('tech-support-dropdown', 'value'),
        State('streaming-tv-dropdown', 'value'),
        State('streaming-movies-dropdown', 'value'),
        State('contract-dropdown', 'value'),
        State('paperless-billing-dropdown', 'value'),
        State('payment-method-dropdown', 'value'),
        State('monthly-charges', 'value'),
        State('total-charges', 'value')
    ]
)

def update_shap_plot(n_clicks, gender, senior, partner, dependents, tenure, phone_service, multiple_lines, internet_service, online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies, contract, paperless_billing, payment_method, monthly_charges, total_charges):
    if n_clicks is not None and n_clicks > 0:
        try:
            tenure = int(tenure)
            monthly_charges = float(monthly_charges)
            total_charges = float(total_charges)
        except ValueError:
            return "Invalid input. Please enter valid numeric values for tenure, monthly charges, and total charges."

        # Create a dictionary with the form data
        form_data = {
            'gender': [gender],
            'SeniorCitizen': [senior],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [int(tenure)],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [float(monthly_charges)],
            'TotalCharges': [float(total_charges)]
        }
        print(form_data)

        # Create a dataframe with all columns
        columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
                   'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                   'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
        data = pd.DataFrame(form_data, columns=columns)

        # Preprocess the data
        preprocessed_data, transformed_columns = ETL.preprocess_data(data)

        # Convert to DataFrame to retain column names
        preprocessed_data = pd.DataFrame(preprocessed_data, columns=transformed_columns)
        # Reorder the columns based on the transformed_columns list
        preprocessed_data = preprocessed_data[transformed_columns]      
        # Import model
        model = load('model.joblib')

        # Make prediction with your model
        prediction = model.predict(preprocessed_data)
        print(preprocessed_data)

        # # Now explain the prediction with SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(preprocessed_data)
        #shap_summary_plot = shap.summary_plot(shap_values, preprocessed_data)
        # Get the absolute SHAP values for each feature
        abs_shap_values = np.abs(shap_values[1][0,:])

        # Get the indices that would sort the SHAP values
        sorted_indices = np.argsort(abs_shap_values)

        # Select the indices of the top N features
        top_n_indices = sorted_indices[-6:]

        # Select only the top N features and SHAP values
        selected_features = preprocessed_data.iloc[0, top_n_indices]
        selected_shap_values = shap_values[1][0, top_n_indices]

        # Create SHAP force plot
        shap.force_plot(explainer.expected_value[1], selected_shap_values, selected_features, feature_names=selected_features.index.tolist(), matplotlib=True, show=False)
        #shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], preprocessed_data.iloc[0,:], feature_names=preprocessed_data.columns.tolist(), matplotlib=True, show=False)

        # Save the plot as an image in memory
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()

        # Encode the image as base64 string
        encoded_image = base64.b64encode(image_png).decode()

        # Display the image as an HTML component
        return html.Img(src='data:image/png;base64,{}'.format(encoded_image), style={'width': '100%'})

    return ''  # Return empty string if n_clicks is None or 0
      

if __name__ == "__main__":
    server.run(debug=True)