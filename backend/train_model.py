import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os

def train_and_export_model():
    print("Loading dataset...")
    data = pd.read_csv('Heart_Disease_Prediction.csv')
    print("Data shape:", data.shape)
    print("Columns:", data.columns.tolist())
    print("Preprocessing data...")
    # Encode target variable
    target_encoder = LabelEncoder()
    data['Heart Disease'] = target_encoder.fit_transform(data['Heart Disease'])
    # Encode categorical variables
    categorical_columns = ['Chest pain type', 'EKG results', 'Slope of ST', 'Thallium']
    for col in categorical_columns:
        if data[col].dtype == object:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
    # Prepare features and target (order must match backend/frontend)
    feature_columns = [
        'Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol',
        'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina',
        'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
    ]
    X = data[feature_columns]
    y = data['Heart Disease']
    print("Feature columns:", X.columns.tolist())
    print("Target distribution:", y.value_counts().to_dict())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Training Logistic Regression model...")
    logistic_model = LogisticRegression(random_state=42)
    logistic_model.fit(X_train_scaled, y_train)
    from sklearn.metrics import accuracy_score, classification_report
    y_pred = logistic_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Absence', 'Presence']))
    print("Exporting model and scaler...")
    joblib.dump(logistic_model, 'heart_disease_model.pkl')
    joblib.dump(scaler, 'heart_disease_scaler.pkl')
    joblib.dump(target_encoder, 'heart_disease_encoder.pkl')
    print("Model exported successfully!")
    print("Files created:")
    print("- heart_disease_model.pkl")
    print("- heart_disease_scaler.pkl") 
    print("- heart_disease_encoder.pkl")
    print("\nTesting exported model...")
    test_features = np.array([[
        55,  # Age
        1,   # Sex
        3,   # Chest pain type
        130, # BP
        245, # Cholesterol
        0,   # FBS over 120
        2,   # EKG results
        150, # Max HR
        0,   # Exercise angina
        0.8, # ST depression
        1,   # Slope of ST
        0,   # Number of vessels fluro
        3    # Thallium
    ]])
    test_features_scaled = scaler.transform(test_features)
    prediction = logistic_model.predict(test_features_scaled)[0]
    prediction_proba = logistic_model.predict_proba(test_features_scaled)[0]
    print(f"Test prediction: {'At Risk' if prediction == 1 else 'Healthy'}")
    print(f"Prediction probabilities: Healthy={prediction_proba[0]:.3f}, At Risk={prediction_proba[1]:.3f}")

if __name__ == "__main__":
    train_and_export_model() 