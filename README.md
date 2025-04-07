# Vehicular Engine Health Prediction Using Deep Learning 🚗🧠

This project predicts the health of a vehicle engine using a hybrid deep learning model combining LSTM and Transformer layers. It analyzes recent engine sensor data to detect potential faults, enabling proactive maintenance decisions.

## 🚀 Features
- Predicts engine health from time-series sensor data
- Hybrid deep learning model (LSTM + Transformer)
- Flask-based web interface for easy CSV uploads
- Displays prediction result as “✅ No Fault” or “🚨 Fault Detected!”

## 🧪 Technologies Used
- Python
- Flask
- TensorFlow / Keras
- Pandas, NumPy
- Scikit-learn
- HTML, CSS

## 📂 Input Format
Upload a CSV file with **exactly 30 rows** and the following columns:
- Lub oil pressure
- Fuel pressure
- Coolant pressure
- Lub oil temp
- Coolant temp
- Engine rpm
- Hour
- Day
- Month

## 📁 File Structure
- `app.py` – Flask app for frontend and prediction
- `main.py` – Deep learning model training and evaluation
- `engine_health_hybrid_model.h5` – Trained hybrid model
- `scaler.pkl` – Scaler used to normalize input data
- `final_model_results_hybrid.json` – Contains accuracy and loss metrics

## 🧠 How It Works
1. Upload a valid CSV file.
2. Data is preprocessed and scaled.
3. Model predicts engine health based on time-series patterns.
4. Result is displayed instantly.

## ⚙️ Run the App
```bash
python app.py
