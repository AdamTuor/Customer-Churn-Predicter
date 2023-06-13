
from dash import Dash, dcc, html
# import dash_core_components as dcc
# import dash_html_components as html
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
# app.layout = app.layout = html.Div([
#     html.H3('Choose your gender:'),
#     dcc.Dropdown(id='gender-dropdown',options=[{'label':'Male','value':'Male'},{'label':'Female','value':'Female'}], placeholder='Select Gender'),
#     html.H3('Are you a senior citizen:'),
#     dcc.Dropdown(id='senior-dropdown',options=[{'label':'Yes','value':1},{'label':'No','value':0}], placeholder='Select Status'),
#     html.H3('Do you have a partner:'),
#     dcc.Dropdown(id='partner-dropdown',options=[{'label':'Yes','value':'Yes'},{'label':'No','value':'No'}], placeholder='Select Partner Status'),
#     html.H3('Do you have dependents:'),
#     dcc.Dropdown(id='dependents-dropdown',options=[{'label':'Yes','value':'Yes'},{'label':'No','value':'No'}], placeholder='Select Dependents Status'),
#     html.H3('Length of tenure(months):'),
#     dcc.Input(id='tenure',type='text',placeholder='Enture tenure(months)'),
#     html.H3('Do you have phone service:'),
#     dcc.Dropdown(id='phone-service-dropdown', options=[{'label': 'Yes', 'value': 'Yes'}, {'label': 'No', 'value': 'No'}], placeholder='Select Phone Service'),
#     html.H3('Do you have multiple lines:'),
#     dcc.Dropdown(id='multiple-lines-dropdown', options=[{'label': 'No phone service', 'value': 'No phone service'},{'label': 'Yes', 'value': 'Yes'}, {'label': 'No', 'value': 'No'}], placeholder='Select Multiple Lines'),
#     html.H3('Do you have internet service:'),
#     dcc.Dropdown(id='internet-service-dropdown', options=[{'label': 'DSL', 'value': 'DSL'},{'label': 'Fiber optic', 'value': 'Fiber optic'}, {'label': 'No', 'value': 'No'}], placeholder='Select Internet Service'),
#     html.H3('The following questions are about internet services, if you selected No above please select No internet service for accuracte results.'),
#     html.H3('Online security:'),
#     dcc.Dropdown(id='online-security-dropdown', options=[{'label': 'Yes', 'value': 'Yes'},{'label': 'No', 'value': 'No'}, {'label': 'No internet service', 'value': 'No internet service'}], placeholder='Select Online Security'),
#     html.H3('Online backup:'),
#     dcc.Dropdown(id='online-backup-dropdown', options=[{'label': 'Yes', 'value': 'Yes'},{'label': 'No', 'value': 'No'}, {'label': 'No internet service', 'value': 'No internet service'}], placeholder='Select Online Backup'),
#     html.H3('Device protection:'),
#     dcc.Dropdown(id='device-protection-dropdown', options=[{'label': 'Yes', 'value': 'Yes'},{'label': 'No', 'value': 'No'}, {'label': 'No internet service', 'value': 'No internet service'}], placeholder='Select Device Protection'),
#     html.H3('Tech support:'),
#     dcc.Dropdown(id='tech-support-dropdown', options=[{'label': 'Yes', 'value': 'Yes'},{'label': 'No', 'value': 'No'}, {'label': 'No internet service', 'value': 'No internet service'}], placeholder='Select Tech Support'),
#     html.H3('Streaming TV:'),
#     dcc.Dropdown(id='streaming-tv-dropdown', options=[{'label': 'Yes', 'value': 'Yes'},{'label': 'No', 'value': 'No'}, {'label': 'No internet service', 'value': 'No internet service'}], placeholder='Select Streaming TV'),
#     html.H3('Streaming movies:'),
#     dcc.Dropdown(id='streaming-movies-dropdown', options=[{'label': 'Yes', 'value': 'Yes'},{'label': 'No', 'value': 'No'}, {'label': 'No internet service', 'value': 'No internet service'}], placeholder='Select Streaming Movies'),
#     html.H3('Contract type:'),
#     dcc.Dropdown(id='contract-dropdown', options=[{'label': 'Month-to-month', 'value': 'Month-to-month'},{'label': 'One year', 'value': 'One year'}, {'label': 'Two year', 'value': 'Two year'}], placeholder='Select Contract Type'),
#     html.H3('Paperless billing:'),
#     dcc.Dropdown(id='paperless-billing-dropdown', options=[{'label': 'Yes', 'value': 'Yes'}, {'label': 'No', 'value': 'No'}], placeholder='Select Paperless Billing'),
#     html.H3('Payment method:'),
#     dcc.Dropdown(id='payment-method-dropdown', options=[{'label': 'Electronic check', 'value': 'Electronic check'},{'label': 'Mailed check', 'value': 'Mailed check'}, {'label': 'Bank transfer (automatic)', 'value': 'Bank transfer (automatic)'},{'label': 'Credit card (automatic)', 'value': 'Credit card (automatic)'}], placeholder='Select Payment Method'),
#     html.H3('Monthly charges:'),
#     dcc.Input(id='monthly-charges',type='text',placeholder='Enture Monthly Charges'),
#     html.H3('Total charges:'),
#     dcc.Input(id='total-charges',type='text',placeholder='Enture Total Charges'),
#     html.Button('Predict', id='predict-button', n_clicks=0),
    
#     html.Div(id='shap-plot')

# ])


app.layout = html.Div(
    style={
        "background-image": 'url("https://www.appier.com/hubfs/Imported_Blog_Media/GettyImages-1030850238-01.jpg")',
        "background-size": "cover",
        "background-position": "center",
        "background-attachment": "fixed",
        
    },
    children=[
        html.H3(
            "Telecom Details Form",
            style={
                "color": "black",
                "font-size": "32px",
                "text-align": "center",
                "margin-top": "10px",
                "padding": "30px 10px",
                "margin-bottom": "10px",
            },
        ),
        html.Div(
            className="form-container",
style={
                "width": "400px",
                "margin": "50px auto",
                "padding": "20px",
                "background-color": "rgba(255, 255, 255, 0.8)",
                "backdrop-filter": "blur(8px)",
                "border-radius": "10px",
                "margin-top": "10px",
},

            children=[
                html.Form(
                    id="telecomForm",
                    children=[

                        html.Label(
                            ["Gender:",html.Br()],
                            htmlFor="gender",
                        ),
                        dcc.Dropdown(
                            id="gender",
                            options=[
                                {"label": "Male", "value": "Male"},
                                {"label": "Female", "value": "Female"},
                            ],style={"width": "100%"},
                        ),
                        html.Label(
                            ["Are you a senior citizen:", html.Br()],
                            htmlFor="SeniorCitizen",
                            ),
                        dcc.Dropdown(
                                id="SeniorCitizen",
                                options=[
                                    {"label": "Yes", "value": "Yes"},
                                    {"label": "No", "value": "No"},
                                ],
                                placeholder="Select Option",
                                style={"width": "100%"},
                            ),

                        html.Label(
                            ["Has a partner?",html.Br()],
                            htmlFor="partner",
                        ),
                        dcc.Dropdown(
                            id="partner",
                            options=[
                                {"label": "Yes", "value": "Yes"},
                                {"label": "No", "value": "No"},
                            ],style={"width": "100%"},
                        ),
                        html.Label(
                            ["Has dependents?",html.Br()],
                            htmlFor="dependents",
                        ),
                        dcc.Dropdown(
                            id="dependents",
                            options=[
                                {"label": "Yes", "value": "Yes"},
                                {"label": "No", "value": "No"},
                            ],style={"width": "100%"},
                        ),
                        html.Label(
                            ["Tenure (in months):",html.Br()],
                            htmlFor="tenure",
                        ),
                        dcc.Input(
                            id="tenure",
                            type="text",style={"width": "100%"},
                        ),
                        html.Label(
                            ["Has phone service?",html.Br()],
                            htmlFor="phoneService",
                        ),
                        dcc.Dropdown(
                            id="phoneService",
                            options=[
                                {"label": "Yes", "value": "Yes"},
                                {"label": "No", "value": "No"},
                            ],style={"width": "100%"},
                        ),
                        html.Label(
                            ["Has multiple lines?",html.Br()],
                            htmlFor="multipleLines",
                        ),
                        dcc.Dropdown(
                            id="multipleLines",
                            options=[
                                {"label": "Yes", "value": "Yes"},
                                {"label": "No", "value": "No"},
                            ],style={"width": "100%"},
                        ),
                        html.Label(
                            ["Type of internet service:",html.Br()],
                            htmlFor="internetService",
                        ),
                        dcc.Dropdown(
                            id="internetService",
                            options=[
                                {"label": "DSL", "value": "DSL"},
                                {"label": "Fiber optic", "value": "Fiber optic"},
                                {"label": "No", "value": "No"},
                            ],style={"width": "100%"},
                        ),
                        html.Label(
                            ["Has online security?",html.Br()],
                            htmlFor="onlineSecurity",
                        ),
                        dcc.Dropdown(
                            id="onlineSecurity",
                            options=[
                                {"label": "Yes", "value": "Yes"},
                                {"label": "No", "value": "No"},
                            ],style={"width": "100%"},
                        ),
                        html.Label(
                            ["Has online backup?",html.Br()],
                            htmlFor="onlineBackup",
                        ),
                        dcc.Dropdown(
                            id="onlineBackup",
                            options=[
                                {"label": "Yes", "value": "Yes"},
                                {"label": "No", "value": "No"},
                            ],style={"width": "100%"},
                        ),
                        html.Label(
                            ["Has device protection?",html.Br()],
                            htmlFor="deviceProtection",
                        ),
                        dcc.Dropdown(
                            id="deviceProtection",
                            options=[
                                {"label": "Yes", "value": "Yes"},
                                {"label": "No", "value": "No"},
                            ],style={"width": "100%"},
                        ),
                        html.Label(
                           [ "Has tech support?",html.Br()],
                            htmlFor="techSupport",
                        ),
                        dcc.Dropdown(
                            id="techSupport",
                            options=[
                                {"label": "Yes", "value": "Yes"},
                                {"label": "No", "value": "No"},
                            ],style={"width": "100%"},
                        ),
                        html.Label(
                            ["Has streaming TV?",html.Br()],
                            htmlFor="streamingTV",
                        ),
                        dcc.Dropdown(
                            id="streamingTV",
                            options=[
                                {"label": "Yes", "value": "Yes"},
                                {"label": "No", "value": "No"},
                            ],style={"width": "100%"},
                        ),
                        html.Label(
                            ["Has streaming movies?",html.Br()],
                            htmlFor="streamingMovies",
                        ),
                        dcc.Dropdown(
                            id="streamingMovies",
                            options=[
                                {"label": "Yes", "value": "Yes"},
                                {"label": "No", "value": "No"},
                            ],style={"width": "100%"},
                        ),
                        html.Label(
                            ["Type of contract:",html.Br()],
                            htmlFor="contract",
                        ),
                        dcc.Dropdown(
                            id="contract",
                            options=[
                                {"label": "Month-to-month", "value": "Month-to-month"},
                                {"label": "One year", "value": "One year"},
                                {"label": "Two year", "value": "Two year"},
                            ],style={"width": "100%"},
                        ),
                        html.Label(
                            ["Paperless billing:", html.Br()],
                            htmlFor="paperless-billing-dropdown",
                        ),
                        dcc.Dropdown(
                            id="paperless-billing-dropdown",
                            options=[
                                {"label": "Yes", "value": "Yes"},
                                {"label": "No", "value": "No"},
                            ],
                            style={"width": "100%"},
                        ),
                        html.Label(
                            ["Payment method:", html.Br()],
                            htmlFor="payment-method-dropdown",
                        ),
                        dcc.Dropdown(
                            id="payment-method-dropdown",
                            options=[
                                {"label": "Electronic check", "value": "Electronic check"},
                                {"label": "Mailed check", "value": "Mailed check"},
                                {"label": "Bank transfer (automatic)", "value": "Bank transfer (automatic)"},
                                {"label": "Credit card (automatic)", "value": "Credit card (automatic)"},
                            ],
                            style={"width": "100%"},
                        ),
                        html.Label(
                            ["Monthly charges:", html.Br()],
                            htmlFor="monthly-charges",
                        ),
                        dcc.Input(
                            id="monthly-charges",
                            type="text",
                            placeholder="Enter Monthly Charges",
                            style={"width": "100%"},
                        ),
                        html.Label(
                            ["Total charges:", html.Br()],
                            htmlFor="total-charges",
                        ),
                        dcc.Input(
                            id="total-charges",
                            type="text",
                            placeholder="Enter Total Charges",
                            style={"width": "100%", "margin-bottom": "10px"},
                        
                        ),
                        html.Button(
                            "Predict",
                            id = "predict-button",
                            type="submit",
                            style={
                            "background-color": "#4CAF50",
                            "color": "black",
                            "width": "100%",
                            "padding": "10px",
                            "margin-top": "10px",
                            "border": "none",
                            "border-radius": "3px",
                            "cursor": "pointer",
                            "font-weight": "bold",
                            "transition": "font-weight 0.3s",
                            "font-size": "16px"
        },
                        ),
                    ],
                ),
            ],
        ),
         html.Div(id="shap-plot")
    ],
)


# Define Dash callback
# @app.callback(
#     Output('shap-plot', 'children'),
#     Input('predict-button', 'n_clicks'),
#     [
#         State('gender', 'value'),
#         State('senior', 'value'),
#         State('partner', 'value'),
#         State('dependents', 'value'),
#         State('tenure', 'value'),
#         State('phoneService', 'value'),
#         State('multipleLines', 'value'),
#         State('internetService', 'value'),
#         State('onlineSecurity', 'value'),
#         State('onlineBackup', 'value'),
#         State('deviceProtection', 'value'),
#         State('techSupport', 'value'),
#         State('streamingTV', 'value'),
#         State('streamingMovies', 'value'),
#         State('contract', 'value'),
#         State('paperless-billing-dropdown', 'value'),
#         State('payment-method-dropdown', 'value'),
#         State('monthly-charges', 'value'),
#         State('total-charges', 'value')
#     ]
# )

# def update_shap_plot(n_clicks, gender, senior, partner, dependents, tenure, phone_service, multiple_lines, internet_service, online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies, contract, paperless_billing_dropdown, payment_method, monthly_charges, total_charges):
#     if n_clicks is not None and n_clicks > 0:
#         try:
#             tenure = int(tenure)
#             monthly_charges = float(monthly_charges)
#             total_charges = float(total_charges)
#         except ValueError:
#             return "Invalid input. Please enter valid numeric values for tenure, monthly charges, and total charges."

#         # Create a dictionary with the form data
#         form_data = {
#             'gender': [gender],
#             'SeniorCitizen': [senior],
#             'Partner': [partner],
#             'Dependents': [dependents],
#             'tenure': [int(tenure)],
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
#             'PaperlessBilling': [paperless_billing_dropdown],
#             'PaymentMethod': [payment_method],
#             'MonthlyCharges': [float(monthly_charges)],
#             'TotalCharges': [float(total_charges)]
#         }
#         print(form_data)

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
#         # Reorder the columns based on the transformed_columns list
#         preprocessed_data = preprocessed_data[transformed_columns]      
#         # Import model
#         model = load('model.joblib')

#         # Make prediction with your model
#         prediction = model.predict(preprocessed_data)
#         print(preprocessed_data)

#         # # Now explain the prediction with SHAP
#         explainer = shap.TreeExplainer(model)
#         shap_values = explainer.shap_values(preprocessed_data)
#         #shap_summary_plot = shap.summary_plot(shap_values, preprocessed_data)
#         # Get the absolute SHAP values for each feature
#         abs_shap_values = np.abs(shap_values[1][0,:])

#         # Get the indices that would sort the SHAP values
#         sorted_indices = np.argsort(abs_shap_values)

#         # Select the indices of the top N features
#         top_n_indices = sorted_indices[-6:]

#         # Select only the top N features and SHAP values
#         selected_features = preprocessed_data.iloc[0, top_n_indices]
#         selected_shap_values = shap_values[1][0, top_n_indices]

#         # Create SHAP force plot
#         shap.force_plot(explainer.expected_value[1], selected_shap_values, selected_features, feature_names=selected_features.index.tolist(), matplotlib=True, show=False)
#         #shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], preprocessed_data.iloc[0,:], feature_names=preprocessed_data.columns.tolist(), matplotlib=True, show=False)

#         # Save the plot as an image in memory
#         buffer = io.BytesIO()
#         plt.savefig(buffer, format='png')
#         buffer.seek(0)
#         image_png = buffer.getvalue()
#         buffer.close()

#         # Encode the image as base64 string
#         encoded_image = base64.b64encode(image_png).decode()

#         # Display the image as an HTML component
#         return html.Img(src='data:image/png;base64,{}'.format(encoded_image), style={'width': '100%'})

#     return ''  # Return empty string if n_clicks is None or 0
      

# if __name__ == "__main__":
#     server.run(debug=True)

@app.callback(
    Output('shap-plot', 'children'),
    Input('predict-button', 'n_clicks'),
    [
        State('gender', 'value'),
        State('senior', 'value'),
        State('partner', 'value'),
        State('dependents', 'value'),
        State('tenure', 'value'),
        State('phoneService', 'value'),
        State('multipleLines', 'value'),
        State('internetService', 'value'),
        State('onlineSecurity', 'value'),
        State('onlineBackup', 'value'),
        State('deviceProtection', 'value'),
        State('techSupport', 'value'),
        State('streamingTV', 'value'),
        State('streamingMovies', 'value'),
        State('contract', 'value'),
        State('paperless-billing-dropdown', 'value'),
        State('payment-method-dropdown', 'value'),
        State('monthly-charges', 'value'),
        State('total-charges', 'value')
    ]
)
def update_shap_plot(n_clicks, gender, senior, partner, dependents, tenure, phone_service, multiple_lines, internet_service, online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies, contract, paperless_billing_dropdown, payment_method, monthly_charges, total_charges):
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
            'PaperlessBilling': [paperless_billing_dropdown],
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

        # Now explain the prediction with SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(preprocessed_data)

        # Get the absolute SHAP values for each feature
        abs_shap_values = np.abs(shap_values[1][0, :])

        # Get the indices that would sort the SHAP values
        sorted_indices = np.argsort(abs_shap_values)

        # Select the indices of the top N features
        top_n_indices = sorted_indices[-6:]

        # Select only the top N features and SHAP values
        selected_features = preprocessed_data.iloc[0, top_n_indices]
        selected_shap_values = shap_values[1][0, top_n_indices]

        # Create SHAP force plot
        shap.force_plot(explainer.expected_value[1], selected_shap_values, selected_features,
                        feature_names=selected_features.index.tolist(), matplotlib=True, show=False)

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
