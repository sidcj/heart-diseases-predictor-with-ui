# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the saved model
model = joblib.load('heart_disease_model.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    cp = int(request.form['cp'])
    thalach = int(request.form['thalach'])
    input_data = np.array([[age, cp, thalach]])
    prediction = model.predict(input_data)
    result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
    return render_template('index.html', prediction_text=f"Prediction: {result}")

if __name__ == '__main__':
    app.run(debug=True)
