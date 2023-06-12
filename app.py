from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd
import json

from sklearn.model_selection import train_test_split
from imblearn.ensemble import EasyEnsembleClassifier

df = pd.read_csv('Resources/Telco-Customer-Churn.csv')

df1 = df.loc[:, ["tenure", 'InternetService', 'Contract', 'PaymentMethod', 'Churn']]

df1.loc[df1["InternetService"]=="DSL", "InternetService"]=1
df1.loc[df1["InternetService"]=="Fiber optic", "InternetService"]=2
df1.loc[df1["InternetService"]=="No", "InternetService"]=3

df1.loc[df1["Contract"]=="One year", "Contract"]=1
df1.loc[df1["Contract"]=="Two year", "Contract"]=2
df1.loc[df1["Contract"]=="Month-to-month", "Contract"]=3

df1.loc[df1["PaymentMethod"]=="Bank transfer (automatic)", "PaymentMethod"]=1
df1.loc[df1["PaymentMethod"]=="Credit card (automatic)", "PaymentMethod"]=2
df1.loc[df1["PaymentMethod"]=="Electronic check", "PaymentMethod"]=3
df1.loc[df1["PaymentMethod"]=="Mailed check", "PaymentMethod"]=3

df1['InternetService'] = df1['InternetService'].astype(int)
df1['Contract'] = df1['Contract'].astype(int)
df1['PaymentMethod'] = df1['PaymentMethod'].astype(int)

y = df1["Churn"]
X = df1.drop("Churn", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.90, random_state=42)

eec = EasyEnsembleClassifier(n_estimators=100, random_state=1)
eec.fit(X_train, y_train)

app = Flask(__name__)

@app.route("/", methods=["GET", 'POST'])
def index():
    return render_template("index.html") 


@app.route("/predict", methods=["GET", 'POST'])
def predict():
    output = "Your value here!"
    #If you have the user submit a form
    if request.method == 'POST':   
        tenure = request.json.get("tenure")
        internetservice = request.json.get("internetservice")
        contract = request.json.get("contract")
        paymentmethod = request.json.get("paymentmethod")
                      
        test_data = pd.DataFrame({'tenure': [tenure], 'internet_service': [internetservice], 'contract': [contract], 
                                  'payment_method': [paymentmethod]})
        
        print("\n============= test_data ====================\n")
        print(test_data)
        print("\n============================================\n")
       
        pred = eec.predict(X_test)
        print("\n============= pred value ====================\n")
        print(pred)
        print("\n=======================================\n")
        
        # if int(pred[0]) == 0:
        #     return {"Prediction": [0]}
        # else:
        #     return {"Prediction": [1]}  
                
        return (json.dumps({'Prediction': (pred[0])}))
        
    return render_template('predictions.html', output=output)

if __name__=="__main__":
    app.run(debug=True)
    

