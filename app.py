from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import os
import json
from werkzeug.utils import secure_filename

# Initialize Flask App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
model_path = os.path.abspath("engine_health_hybrid_model.h5")
scaler_path = os.path.abspath("scaler.pkl")

model = load_model(model_path)
scaler = joblib.load(scaler_path)  # Load the saved MinMaxScaler

# Define Expected Features
features = ['Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 
            'lub oil temp', 'Coolant temp', 'Engine rpm', 'Hour', 'Day', 'Month']

# Load Model Accuracies
def load_accuracies():
    accuracy_file = "final_model_results_hybrid.json"
    if os.path.exists(accuracy_file):
        with open(accuracy_file, "r") as f:
            return json.load(f)
    return {}

# Convert Probability to Label
def get_prediction_label(value):
    return "🚨 Fault Detected!" if value == 1 else "✅ No Fault"

# Function to Preprocess Input CSV
def preprocess_input_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        if len(df) != 30:
            return "CSV file must contain exactly 30 rows of data."
        
        df_scaled = scaler.transform(df[features])
        sequence_input = np.expand_dims(df_scaled, axis=0)
        return sequence_input
    except Exception as e:
        return str(e)

# Home Route - Display Accuracies
@app.route('/')
def home():
    accuracies = load_accuracies()
    return render_template('home.html', accuracies=accuracies)

# Prediction Route - Accepts CSV Upload
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('predict.html', error="No file uploaded.")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('predict.html', error="No selected file.")
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(file_path)
        
        input_data = preprocess_input_csv(file_path)
        if isinstance(input_data, str):
            return render_template('predict.html', error=input_data)
        
        probability = model.predict(input_data)[0][0]
        prediction = int(probability > 0.5)
        prediction_label = get_prediction_label(prediction)
        
        return render_template('predict.html', prediction=prediction_label)
    
    return render_template('predict.html', prediction=None)

# Run the Flask App
if __name__ == '__main__':
    app.run(debug=True)