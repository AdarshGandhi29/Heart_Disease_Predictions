import pandas as pd
import joblib
import numpy as np

def load_model_and_data():
    """Load the trained model, scaler, and dataset"""
    try:
        model = joblib.load('heart_disease_model.pkl')
        scaler = joblib.load('heart_disease_scaler.pkl')
        df = pd.read_csv('Heart_Disease_Prediction_Dataset.csv')
        print("Model and data loaded successfully!")
        return model, scaler, df
    except Exception as e:
        print(f"Error loading model or data: {e}")
        return None, None, None

def find_sample_entries():
    """Find sample entries that predict 'no heart disease' for frontend testing"""
    model, scaler, df = load_model_and_data()
    
    if model is None or scaler is None or df is None:
        return
    
    # Prepare features
    feature_columns = [
        'Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 
        'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina', 
        'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
    ]
    
    X = df[feature_columns].values
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    prediction_probas = model.predict_proba(X_scaled)
    
    # Find entries where model predicts absence (0)
    absence_indices = np.where(predictions == 0)[0]
    
    print("=== SAMPLE ENTRIES FOR 'NO HEART DISEASE' PREDICTION ===\n")
    print("Use these values in the frontend form to get 'No Heart Disease' predictions:\n")
    
    # Get top 10 entries with highest confidence for absence
    absence_entries = df.iloc[absence_indices].copy()
    absence_entries['confidence'] = [prediction_probas[i][0] for i in absence_indices]
    absence_entries = absence_entries.sort_values('confidence', ascending=False)
    
    for i, (idx, row) in enumerate(absence_entries.head(10).iterrows(), 1):
        print(f"Sample Entry #{i} (Confidence: {row['confidence']:.3f}):")
        print("=" * 50)
        print(f"Age: {row['Age']}")
        print(f"Sex: {row['Sex']} (0=Female, 1=Male)")
        print(f"Chest Pain Type: {row['Chest pain type']} (0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic)")
        print(f"BP: {row['BP']}")
        print(f"Cholesterol: {row['Cholesterol']}")
        print(f"FBS over 120: {row['FBS over 120']} (0=No, 1=Yes)")
        print(f"EKG Results: {row['EKG results']} (0=Normal, 1=ST-T, 2=LVH)")
        print(f"Max HR: {row['Max HR']}")
        print(f"Exercise Angina: {row['Exercise angina']} (0=No, 1=Yes)")
        print(f"ST Depression: {row['ST depression']}")
        print(f"Slope of ST: {row['Slope of ST']} (0=Upsloping, 1=Flat, 2=Downsloping)")
        print(f"Number of Vessels: {row['Number of vessels fluro']} (0-3)")
        print(f"Thallium: {row['Thallium']} (1=Normal, 2=Fixed, 3=Reversible)")
        print(f"Actual Label: {row['Heart Disease']}")
        print()
    
    # Also provide some simple, healthy-looking examples
    print("=== SIMPLE HEALTHY EXAMPLES ===")
    print("These are examples with typical healthy values:")
    print()
    
    # Find entries with typical healthy values
    healthy_examples = absence_entries[
        (absence_entries['Age'] < 60) &
        (absence_entries['BP'] < 140) &
        (absence_entries['Cholesterol'] < 250) &
        (absence_entries['Max HR'] > 140)
    ].head(5)
    
    for i, (idx, row) in enumerate(healthy_examples.iterrows(), 1):
        print(f"Healthy Example #{i}:")
        print(f"Age: {row['Age']}, Sex: {row['Sex']}, BP: {row['BP']}, Cholesterol: {row['Cholesterol']}")
        print(f"Max HR: {row['Max HR']}, Chest Pain: {row['Chest pain type']}")
        print(f"Confidence: {row['confidence']:.3f}")
        print()

if __name__ == "__main__":
    find_sample_entries() 