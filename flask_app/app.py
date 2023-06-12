# from flask import Flask, render_template, request
# import pandas as pd
# import flask_app.ETL as ETL
# from sklearn.externals import joblib
# import shap

# app = Flask(__name__)
# # Import model
# model = joblib.load('model.joblib')

# @app.route('/')
# def index():
#     return render_template('form.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Extract form data
#     form_data = request.form

#     # Convert form data to dataframe
#     data = pd.DataFrame([form_data])

#     # Preprocess the data
#     preprocessed_data, transformed_columns = ETL.preprocess_data(data)

#     # Convert to DataFrame to retain column names
#     preprocessed_data = pd.DataFrame(preprocessed_data, columns=transformed_columns)

#     # Make prediction with your model
#     prediction = model.predict(preprocessed_data)

#     # Now explain the prediction with SHAP
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(preprocessed_data)

#     # Get SHAP values for the first prediction
#     shap_values_for_prediction = shap_values[0]

#     # Render a new template and pass prediction and SHAP values
#     return render_template('prediction.html', prediction=prediction[0], shap_values=shap_values_for_prediction)

from flask import Flask, request
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

# Initialize Flask server
server = Flask(__name__)

# Initialize Dash app
app = Dash(__name__, server=server)

# Define Dash layout
app.layout = app.layout = html.Div([
    dcc.Dropdown(id='gender-dropdown',options=[{'label':'Male','value':'Male'},{'label':'Female','value':'Female'}], placeholder='Select Gender'),
    dcc.Dropdown(id='senior-dropdown',options=[{'label':'Yes','value':1},{'label':'No','value':0}], placeholder='Select Status'),
    dcc.Dropdown(id='partner-dropdown',options=[{'label':'Yes','value':'Yes'},{'label':'No','value':'No'}], placeholder='Select Partner Status'),
    dcc.Dropdown(id='dependents-dropdown',options=[{'label':'Yes','value':'Yes'},{'label':'No','value':'No'}], placeholder='Select Dependents Status'),
    dcc.Input(id='tenure',type='text',placeholder='Enture tenure(months)'),
    dcc.Dropdown(id='phone-service-dropdown', options=[{'label': 'Yes', 'value': 'Yes'}, {'label': 'No', 'value': 'No'}], placeholder='Select Phone Service'),
    dcc.Dropdown(id='multiple-lines-dropdown', options=[{'label': 'No phone service', 'value': 'No phone service'},{'label': 'Yes', 'value': 'Yes'}, {'label': 'No', 'value': 'No'}], placeholder='Select Multiple Lines'),
    dcc.Dropdown(id='internet-service-dropdown', options=[{'label': 'DSL', 'value': 'DSL'},{'label': 'Fiber optic', 'value': 'Fiber Optic'}, {'label': 'No', 'value': 'No'}], placeholder='Select Internet Service'),
    dcc.Dropdown(id='online-security-dropdown', options=[{'label': 'Yes', 'value': 'Yes'},{'label': 'No', 'value': 'No'}, {'label': 'No internet service', 'value': 'No internet service'}], placeholder='Select Online Security'),
    dcc.Dropdown(id='online-backup-dropdown', options=[{'label': 'Yes', 'value': 'Yes'},{'label': 'No', 'value': 'No'}, {'label': 'No internet service', 'value': 'No internet service'}], placeholder='Select Online Backup'),
    dcc.Dropdown(id='device-protection-dropdown', options=[{'label': 'Yes', 'value': 'Yes'},{'label': 'No', 'value': 'No'}, {'label': 'No internet service', 'value': 'No internet service'}], placeholder='Select Device Protection'),
    dcc.Dropdown(id='tech-support-dropdown', options=[{'label': 'Yes', 'value': 'Yes'},{'label': 'No', 'value': 'No'}, {'label': 'No internet service', 'value': 'No internet service'}], placeholder='Select Tech Support'),
    dcc.Dropdown(id='streaming-tv-dropdown', options=[{'label': 'Yes', 'value': 'Yes'},{'label': 'No', 'value': 'No'}, {'label': 'No internet service', 'value': 'No internet service'}], placeholder='Select Streaming TV'),
    dcc.Dropdown(id='streaming-movies-dropdown', options=[{'label': 'Yes', 'value': 'Yes'},{'label': 'No', 'value': 'No'}, {'label': 'No internet service', 'value': 'No internet service'}], placeholder='Select Streaming Movies'),
    dcc.Dropdown(id='contract-dropdown', options=[{'label': 'Month-to-month', 'value': 'Month-to-month'},{'label': 'One year', 'value': 'One year'}, {'label': 'Two year', 'value': 'Two year'}], placeholder='Select Contract Type'),
    dcc.Dropdown(id='paperless-billing-dropdown', options=[{'label': 'Yes', 'value': 'Yes'}, {'label': 'No', 'value': 'No'}], placeholder='Select Paperless Billing'),
    dcc.Dropdown(id='payment-method-dropdown', options=[{'label': 'Electronic check', 'value': 'Electronic check'},{'label': 'Mailed check', 'value': 'Mailed check'}, {'label': 'Bank transfer (automatic)', 'value': 'Bank transfer (automatic)'},{'label': 'Credit card (automatic)', 'value': 'Credit card (automatic)'}], placeholder='Select Payment Method'),
    dcc.Input(id='monthly-charges',type='text',placeholder='Enture Monthly Charges'),
    dcc.Input(id='total-charges',type='text',placeholder='Enture Total Charges'),
    html.Button('Predict', id='predict-button', n_clicks=0),
    
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
# def update_output(n_clicks, gender, senior, partner, dependents, tenure, phone_service, multiple_lines, internet_service, online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies, contract, paperless_billing, payment_method, monthly_charges, total_charges):
#     if n_clicks is not None and n_clicks > 0:
#         # Create a dictionary with the form data
#         form_data = {
#             'gender': gender,
#             'SeniorCitizen': senior,
#             'Partner': partner,
#             'Dependents': dependents,
#             'tenure': tenure,
#             'PhoneService': phone_service,
#             'MultipleLines': multiple_lines,
#             'InternetService': internet_service,
#             'OnlineSecurity': online_security,
#             'OnlineBackup': online_backup,
#             'DeviceProtection': device_protection,
#             'TechSupport': tech_support,
#             'StreamingTV': streaming_tv,
#             'StreamingMovies': streaming_movies,
#             'Contract': contract,
#             'PaperlessBilling': paperless_billing,
#             'PaymentMethod': payment_method,
#             'MonthlyCharges': monthly_charges,
#             'TotalCharges': total_charges
#         }

#         # Convert form data to dataframe
#         data = pd.DataFrame([form_data])

#         # Preprocess the data
#         preprocessed_data, transformed_columns = ETL.preprocess_data(data)

#         # Convert to DataFrame to retain column names
#         preprocessed_data = pd.DataFrame(preprocessed_data, columns=transformed_columns)
#         print(preprocessed_data)

#         # Import model
#         model = load('model.joblib')

#         # Make prediction with your model
#         prediction = model.predict(preprocessed_data)

#         # Now explain the prediction with SHAP
#         explainer = shap.TreeExplainer(model)
#         shap_values = explainer.shap_values(preprocessed_data)

#         # Get SHAP values for the first prediction
#         shap_values_for_prediction = shap_values[0]

#     # Return the prediction and SHAP values
#     return f'Prediction: {prediction[0]}, SHAP values: {shap_values_for_prediction}'
# def update_output(n_clicks, gender, senior, partner, dependents, tenure, phone_service, multiple_lines, internet_service, online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies, contract, paperless_billing, payment_method, monthly_charges, total_charges):
#     if n_clicks is not None and n_clicks > 0:
#         # Create a dictionary with the form data
#         form_data = {
#             'gender': [gender],
#             'SeniorCitizen': [senior],
#             'Partner': [partner],
#             'Dependents': [dependents],
#             'tenure': [tenure],
#             'PhoneService': [phone_service],
#             'MultipleLines': [multiple_lines],
#             'InternetService': [internet_service],
#             'OnlineSecurity': [online_security],
#             'OnlineBackup': [online_backup],
#             'DeviceProtection': [device_protection],
#             'TechSupport': [tech_support],
#             'StreamingTV': [streaming_tv],
#             'StreamingMovies': [streaming_movies],
#             'Contract': [contract],
#             'PaperlessBilling': [paperless_billing],
#             'PaymentMethod': [payment_method],
#             'MonthlyCharges': [monthly_charges],
#             'TotalCharges': [total_charges]
#         }

#         # Create a dataframe with all columns
#         columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
#                    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
#                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
#                    'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
#         data = pd.DataFrame(form_data, columns=columns)

#         # Preprocess the data
#         preprocessed_data, transformed_columns = ETL.preprocess_data(data)

#         # Convert to DataFrame to retain column names
#         preprocessed_data = pd.DataFrame(preprocessed_data, columns=transformed_columns)

#         # Print the shape and columns of the preprocessed data
#         print("Preprocessed Data Shape:", preprocessed_data.shape)
#         print("Preprocessed Data Columns:", preprocessed_data.columns)
#         # Import model
#         model = load('model.joblib')

#         # Make prediction with your model
#         prediction = model.predict(preprocessed_data)

#         # Now explain the prediction with SHAP
#         explainer = shap.TreeExplainer(model)
#         shap_values = explainer.shap_values(preprocessed_data)
#         shap_plot = shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], preprocessed_data.iloc[0,:], feature_names=preprocessed_data.columns.tolist())
#         shap_graph = dcc.Graph(figure=shap_plot)
#         # Get SHAP values for the first prediction
#         #shap_values_for_prediction = shap_values[0]
#         return shap_graph
#         # Return the prediction and SHAP values
#         #return f'Prediction: {prediction[0]}, SHAP values: {shap_values_for_prediction}'

#     return ''  # Return empty string if n_clicks is None or 0
# # Run server

def update_shap_plot(n_clicks, gender, senior, partner, dependents, tenure, phone_service, multiple_lines, internet_service, online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies, contract, paperless_billing, payment_method, monthly_charges, total_charges):
    if n_clicks is not None and n_clicks > 0:
        # Create a dictionary with the form data
        form_data = {
            'gender': [gender],
            'SeniorCitizen': [senior],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
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
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        }

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

        # Import model
        model = load('model.joblib')

        # Make prediction with your model
        prediction = model.predict(preprocessed_data)
        print(preprocessed_data)

        # # Now explain the prediction with SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(preprocessed_data)
        #shap_summary_plot = shap.summary_plot(shap_values, preprocessed_data)

        # Create SHAP force plot
        shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], preprocessed_data.iloc[0,:], feature_names=preprocessed_data.columns.tolist(), matplotlib=True, show=False)

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
        # return [
        #     html.Div(f'Prediction: {prediction[0]}'),
        #     shap_graph
        # ]

    

# Run server
if __name__ == "__main__":
    server.run(debug=True)
