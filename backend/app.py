from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Global variables to store the model and scaler
model = None
scaler = None
label_encoder = None

def load_model():
    """Load the trained model and scaler"""
    global model, scaler, label_encoder
    
    try:
        # Load the best performing model (Logistic Regression)
        model = joblib.load('heart_disease_model.pkl')
        scaler = joblib.load('heart_disease_scaler.pkl')
        label_encoder = joblib.load('heart_disease_encoder.pkl')
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Model files not found. Please run the model training script first.")
        return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    return True

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Heart Disease Prediction API is running'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict heart disease risk based on input features"""
    
    # Check if model is loaded
    if model is None or scaler is None:
        print('Model or scaler not loaded!')
        return jsonify({
            'error': 'Model not loaded. Please ensure the model files are available.'
        }), 500
    
    try:
        # Get input data
        data = request.get_json()
        print('--- Received data from frontend ---')
        print(data)
        
        if not data:
            print('No input data provided')
            return jsonify({
                'error': 'No input data provided'
            }), 400
        
        # Validate required fields
        required_fields = [
            'Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol',
            'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina',
            'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            print(f'Missing required fields: {missing_fields}')
            return jsonify({
                'error': f'Missing required fields: {missing_fields}'
            }), 400
        
        # Create feature array in the correct order
        features = np.array([
            data['Age'],
            data['Sex'],
            data['Chest pain type'],
            data['BP'],
            data['Cholesterol'],
            data['FBS over 120'],
            data['EKG results'],
            data['Max HR'],
            data['Exercise angina'],
            data['ST depression'],
            data['Slope of ST'],
            data['Number of vessels fluro'],
            data['Thallium']
        ]).reshape(1, -1)
        print('--- Feature array (before scaling) ---')
        print(features)
        # Scale the features
        features_scaled = scaler.transform(features)
        print('--- Feature array (after scaling) ---')
        print(features_scaled)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        # Convert prediction to human-readable format
        prediction_label = "At Risk" if prediction == 1 else "Healthy"
        confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]
        
        return jsonify({
            'prediction': prediction_label,
            'confidence': round(confidence * 100, 2),
            'prediction_probability': {
                'Healthy': round(prediction_proba[0] * 100, 2),
                'At Risk': round(prediction_proba[1] * 100, 2)
            },
            'input_features': data
        })
        
    except ValueError as e:
        print(f'Invalid input data: {str(e)}')
        return jsonify({
            'error': f'Invalid input data: {str(e)}'
        }), 400
    except Exception as e:
        print(f'Prediction error: {str(e)}')
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 500
    
    return jsonify({
        'model_type': type(model).__name__,
        'model_accuracy': '92.59% (from training)',
        'features': [
            'Age', 'Sex', 'Chest_pain_type', 'BP', 'Cholesterol', 
            'FBS_over_120', 'EKG_results', 'Max_HR', 'Exercise_angina', 
            'ST_depression', 'Slope_of_ST', 'Number_of_vessels_fluro', 'Thallium'
        ],
        'feature_count': 13
    })

form_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Heart Disease Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f7f7f7; }
        .container { max-width: 500px; margin: 40px auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px #ccc; }
        h2 { text-align: center; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; }
        input, select { width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; }
        button { width: 100%; padding: 10px; background: #007bff; color: #fff; border: none; border-radius: 4px; font-size: 16px; }
        .result { margin-top: 20px; padding: 15px; border-radius: 4px; background: #e9ecef; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Heart Disease Prediction</h2>
        <form method="post">
            {% for field in fields %}
            <div class="form-group">
                <label for="{{ field }}">{{ field.replace('_', ' ') }}</label>
                <input type="number" step="any" name="{{ field }}" id="{{ field }}" required>
            </div>
            {% endfor %}
            <button type="submit">Predict</button>
        </form>
        {% if result %}
        <div class="result">
            <strong>Prediction:</strong> {{ result['prediction'] }}<br>
            <strong>Confidence:</strong> {{ result['confidence'] }}%<br>
            <strong>Probabilities:</strong> Healthy: {{ result['prediction_probability']['Healthy'] }}%, At Risk: {{ result['prediction_probability']['At Risk'] }}%
        </div>
        {% elif error %}
        <div class="result" style="background:#f8d7da; color:#721c24;">
            <strong>Error:</strong> {{ error }}
        </div>
        {% endif %}
    </div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def web_form():
    fields = [
        'Age', 'Sex', 'Chest_pain_type', 'BP', 'Cholesterol',
        'FBS_over_120', 'EKG_results', 'Max_HR', 'Exercise_angina',
        'ST_depression', 'Slope_of_ST', 'Number_of_vessels_fluro', 'Thallium'
    ]
    result = None
    error = None
    if request.method == 'POST':
        try:
            # Collect and validate input
            data = {}
            for field in fields:
                value = request.form.get(field)
                if value is None or value == '':
                    raise ValueError(f"Missing value for {field}")
                # Convert to float or int as appropriate
                if field in ['ST_depression']:
                    data[field] = float(value)
                else:
                    data[field] = int(float(value))
            # Prepare features
            features = np.array([
                data['Age'],
                data['Sex'],
                data['Chest_pain_type'],
                data['BP'],
                data['Cholesterol'],
                data['FBS_over_120'],
                data['EKG_results'],
                data['Max_HR'],
                data['Exercise_angina'],
                data['ST_depression'],
                data['Slope_of_ST'],
                data['Number_of_vessels_fluro'],
                data['Thallium']
            ]).reshape(1, -1)
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            prediction_proba = model.predict_proba(features_scaled)[0]
            prediction_label = "At Risk" if prediction == 1 else "Healthy"
            confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]
            result = {
                'prediction': prediction_label,
                'confidence': round(confidence * 100, 2),
                'prediction_probability': {
                    'Healthy': round(prediction_proba[0] * 100, 2),
                    'At Risk': round(prediction_proba[1] * 100, 2)
                }
            }
        except Exception as e:
            error = str(e)
    return render_template_string(form_html, fields=fields, result=result, error=error)

if __name__ == '__main__':
    # Load the model when starting the app
    if load_model():
        print("Starting Heart Disease Prediction API...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please check if model files exist.") 