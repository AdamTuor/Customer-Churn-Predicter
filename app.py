from flask import Flask, render_template
import csv

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return 'Welcome to the API'

@app.route('/data', methods=['GET'])
def get_data():
    data = []
    with open(r'C:\Users\tehma\OneDrive\Desktop\project-4-main\Resources\cleaned_telecom_data.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return render_template('data.html', data=data)

if __name__ == '__main__':
    app.run()
