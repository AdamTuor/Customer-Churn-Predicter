import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

def generate_fake_data(num_samples):
    data = []
    for _ in range(num_samples):
        phone_service = random.choice(['Yes', 'No'])
        if phone_service == 'Yes':
            multiple_lines = random.choice(['Yes', 'No'])
        else:
            multiple_lines = 'No phone service'

        internet_service = random.choice(['DSL', 'Fiber optic']) if phone_service == 'No' else random.choice(['DSL', 'Fiber optic', 'No'])

        sample = {
            'gender': random.choice(['Female', 'Male']),
            'SeniorCitizen': random.choice([0, 1]),
            'Partner': random.choice(['Yes', 'No']),
            'Dependents': random.choice(['Yes', 'No']),
            'tenure': random.randint(0, 72),
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': 'No internet service' if internet_service == 'No' else random.choice(['Yes', 'No']),
            'OnlineBackup': 'No internet service' if internet_service == 'No' else random.choice(['Yes', 'No']),
            'DeviceProtection': 'No internet service' if internet_service == 'No' else random.choice(['Yes', 'No']),
            'TechSupport': 'No internet service' if internet_service == 'No' else random.choice(['Yes', 'No']),
            'StreamingTV': 'No internet service' if internet_service == 'No' else random.choice(['Yes', 'No']),
            'StreamingMovies': 'No internet service' if internet_service == 'No' else random.choice(['Yes', 'No']),
            'Contract': random.choice(['Month-to-month', 'One year', 'Two year']),
            'PaperlessBilling': random.choice(['Yes', 'No']),
            'PaymentMethod': random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']),
            'MonthlyCharges': random.uniform(18.25, 118.75),
        }
        # If tenure is 0, assume they paid for their first month.
        sample['TotalCharges'] = (sample['tenure'] + 1) * sample['MonthlyCharges']

        data.append(sample)
    
    return pd.DataFrame(data)

def plot_churn(df, feature, target='Churn'):
    plt.figure(figsize=(10,6))
    sns.countplot(data=df, x=feature, hue=target)
    plt.title(f'Churn by {feature}', fontsize=15)
    plt.ylabel('Count', fontsize=12)
    plt.xlabel(feature, fontsize=12)
    plt.xticks(rotation=45)
    plt.show()