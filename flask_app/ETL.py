from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
from joblib import load
import numpy as np
from sklearn.compose import make_column_transformer

encoder = load('encoder.joblib')
scaler = load('scaler.joblib')
# def preprocess_data(data):
#     print(type(scaler))
#     # Define preprocessing for numerical columns (scale them)
#     numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
#     numerical_transformer = scaler

#     # Define preprocessing for categorical columns (encode them)
#     categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
#                             'PhoneService', 'MultipleLines', 'InternetService',
#                             'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
#                             'TechSupport', 'StreamingTV', 'StreamingMovies',
#                             'Contract', 'PaperlessBilling', 'PaymentMethod']

#     # Add all possible values for each categorical feature
#     possible_values = {
#         'gender': ['Female', 'Male'],
#         'SeniorCitizen': [0, 1],
#         'Partner': ['No', 'Yes'],  # Swapped the order to match training dataset
#         'Dependents': ['No', 'Yes'],  # Swapped the order to match training dataset
#         'PhoneService': ['No', 'Yes'],  # Swapped the order to match training dataset
#         'MultipleLines': ['No','No phone service', 'Yes'],
#         'InternetService': ['DSL', 'Fiber optic', 'No'],
#         'OnlineSecurity': ['No', 'No internet service','Yes'],
#         'OnlineBackup': ['No', 'No internet service','Yes'],
#         'DeviceProtection': ['No', 'No internet service','Yes'],
#         'TechSupport': ['No', 'No internet service','Yes'],
#         'StreamingTV': ['No', 'No internet service','Yes'],
#         'StreamingMovies': ['No', 'No internet service','Yes'],
#         'Contract': ['Month-to-month', 'One year', 'Two year'],
#         'PaperlessBilling': ['No', 'Yes'],
#         'PaymentMethod': ['Bank transfer (automatic)', 'Credit card (automatic)','Electronic check', 'Mailed check']
#     }

#     # Define preprocessing for categorical columns (encode them)
#     categorical_transformer = encoder
#     #OneHotEncoder(categories=[possible_values[feature] for feature in categorical_features])

#     # Combine preprocessing steps
#     # preprocessor = make_column_transformer(
#     #     (numerical_transformer, numerical_features),
#     #     (categorical_transformer, categorical_features)
#     # )
#     preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numerical_transformer, numerical_features),
#         ('cat', categorical_transformer, categorical_features)])

#     # Create a preprocessing and training pipeline
#     pipeline = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#     ])
#     data = pd.DataFrame(data, columns=numerical_features + categorical_features)
#     # Fit the pipeline on the training data
#     #pipeline.fit(data)

#     # Transform the new data using the fitted pipeline
#     processed_data = pipeline.transform(data)

#     # Getting feature names
#     transformed_columns = numerical_features + list(pipeline.named_steps['preprocessor'].named_transformers_['onehotencoder'].get_feature_names_out(categorical_features))

#     # Return processed data and transformed column names
#     return processed_data, transformed_columns
def preprocess_data(data):
    # Load the saved scaler and encoder
    scaler = load('scaler.joblib')
    encoder = load('encoder.joblib')
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                             'PhoneService', 'MultipleLines', 'InternetService',
                             'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                             'TechSupport', 'StreamingTV', 'StreamingMovies',
                             'Contract', 'PaperlessBilling', 'PaymentMethod']
    
    # Scale the numerical features
    scaled_features = scaler.transform(data[numerical_features])
    # One-hot encode the categorical features
    encoded_features = encoder.transform(data[categorical_features])

    # Combine the features
    processed_data = np.hstack((scaled_features, encoded_features.toarray()))

    # Getting feature names
    transformed_columns = numerical_features + list(encoder.get_feature_names_out(categorical_features))

    # Return processed data and transformed column names
    return processed_data, transformed_columns
