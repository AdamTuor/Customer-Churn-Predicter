from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

from sklearn.compose import make_column_transformer

# def preprocess_data(data):
#     # Define preprocessing for numerical columns (scale them)
#     numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
#     numerical_transformer = StandardScaler()

#     # Define preprocessing for categorical columns (encode them)
#     categorical_features = ['gender', 
#               'SeniorCitizen', 
#               'Partner', 
#               'Dependents',
#               'PhoneService', 
#               'MultipleLines', 
#               'InternetService', 
#               'OnlineSecurity',
#               'OnlineBackup', 
#               'DeviceProtection', 
#               'TechSupport', 
#               'StreamingTV',
#               'StreamingMovies', 
#               'Contract', 
#               'PaperlessBilling', 
#               'PaymentMethod']
#     categorical_transformer = OneHotEncoder()

#     # Combine preprocessing steps
#     preprocessor = make_column_transformer(
#         (numerical_transformer, numerical_features),
#         (categorical_transformer, categorical_features)
#     )

#     # Create a preprocessing and training pipeline
#     pipeline = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#     ])

#     # Fit and Transform data
#     processed_data = pipeline.fit_transform(data)

#     # Getting feature names
#     transformed_columns = numerical_features + list(pipeline.named_steps['preprocessor'].named_transformers_['onehotencoder'].get_feature_names_out(categorical_features))

#     # Return processed data and transformed column names
#     return processed_data, transformed_columns

def preprocess_data(data):
    # Define preprocessing for numerical columns (scale them)
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    numerical_transformer = StandardScaler()

    # Define preprocessing for categorical columns (encode them)
    categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                            'PhoneService', 'MultipleLines', 'InternetService',
                            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                            'TechSupport', 'StreamingTV', 'StreamingMovies',
                            'Contract', 'PaperlessBilling', 'PaymentMethod']

# Add all possible values for each categorical feature
    possible_values = {
        'gender': ['Female', 'Male'],
        'SeniorCitizen': [0, 1],
        'Partner': ['Yes', 'No'],
        'Dependents': ['Yes', 'No'],
        'PhoneService': ['Yes', 'No'],
        'MultipleLines': ['No phone service', 'No', 'Yes'],
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'OnlineSecurity': ['No', 'Yes', 'No internet service'],
        'OnlineBackup': ['No', 'Yes', 'No internet service'],
        'DeviceProtection': ['No', 'Yes', 'No internet service'],
        'TechSupport': ['No', 'Yes', 'No internet service'],
        'StreamingTV': ['No', 'Yes', 'No internet service'],
        'StreamingMovies': ['No', 'Yes', 'No internet service'],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'PaperlessBilling': ['Yes', 'No'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
    }

    # Define preprocessing for categorical columns (encode them)
    categorical_transformer = OneHotEncoder(categories=[possible_values[feature] for feature in categorical_features])

    # Combine preprocessing steps
    preprocessor = make_column_transformer(
        (numerical_transformer, numerical_features),
        (categorical_transformer, categorical_features)
    )

    # Create a preprocessing and training pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
    ])

    # Fit the pipeline on the training data
    pipeline.fit(data)

    # Transform the new data using the fitted pipeline
    processed_data = pipeline.transform(data)

    # Getting feature names
    transformed_columns = numerical_features + list(pipeline.named_steps['preprocessor'].named_transformers_['onehotencoder'].get_feature_names_out(categorical_features))

    # Return processed data and transformed column names
    return processed_data, transformed_columns
