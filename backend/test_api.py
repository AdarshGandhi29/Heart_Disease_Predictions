import requests
import json
import time

def test_api():
    """Test the heart disease prediction API"""
    
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Heart Disease Prediction API")
    print("=" * 50)
    
    # Test 1: Health Check
    print("\n1. Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running.")
        return
    
    # Test 2: Model Info
    print("\n2. Testing Model Info...")
    try:
        response = requests.get(f"{base_url}/model_info")
        if response.status_code == 200:
            print("✅ Model info retrieved")
            model_info = response.json()
            print(f"Model Type: {model_info['model_type']}")
            print(f"Accuracy: {model_info['model_accuracy']}")
            print(f"Features: {len(model_info['features'])}")
        else:
            print(f"❌ Model info failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Model info error: {e}")
    
    # Test 3: Prediction with healthy patient
    print("\n3. Testing Prediction (Healthy Patient)...")
    healthy_patient = {
        "Age": 45,
        "Sex": 0,  # Female
        "Chest_pain_type": 1,
        "BP": 120,
        "Cholesterol": 200,
        "FBS_over_120": 0,
        "EKG_results": 0,
        "Max_HR": 160,
        "Exercise_angina": 0,
        "ST_depression": 0.0,
        "Slope_of_ST": 1,
        "Number_of_vessels_fluro": 0,
        "Thallium": 3
    }
    
    try:
        response = requests.post(f"{base_url}/predict", json=healthy_patient)
        if response.status_code == 200:
            print("✅ Prediction successful")
            result = response.json()
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']}%")
            print(f"Probabilities: {result['prediction_probability']}")
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"Error: {response.json()}")
    except Exception as e:
        print(f"❌ Prediction error: {e}")
    
    # Test 4: Prediction with at-risk patient
    print("\n4. Testing Prediction (At-Risk Patient)...")
    at_risk_patient = {
        "Age": 65,
        "Sex": 1,  # Male
        "Chest_pain_type": 4,
        "BP": 180,
        "Cholesterol": 350,
        "FBS_over_120": 1,
        "EKG_results": 2,
        "Max_HR": 120,
        "Exercise_angina": 1,
        "ST_depression": 2.5,
        "Slope_of_ST": 3,
        "Number_of_vessels_fluro": 3,
        "Thallium": 7
    }
    
    try:
        response = requests.post(f"{base_url}/predict", json=at_risk_patient)
        if response.status_code == 200:
            print("✅ Prediction successful")
            result = response.json()
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']}%")
            print(f"Probabilities: {result['prediction_probability']}")
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"Error: {response.json()}")
    except Exception as e:
        print(f"❌ Prediction error: {e}")
    
    # Test 5: Invalid input (missing fields)
    print("\n5. Testing Invalid Input (Missing Fields)...")
    invalid_data = {
        "Age": 50,
        "Sex": 1
        # Missing other required fields
    }
    
    try:
        response = requests.post(f"{base_url}/predict", json=invalid_data)
        if response.status_code == 400:
            print("✅ Invalid input properly handled")
            print(f"Error: {response.json()}")
        else:
            print(f"❌ Expected 400 error, got: {response.status_code}")
    except Exception as e:
        print(f"❌ Invalid input test error: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 API Testing Complete!")

if __name__ == "__main__":
    test_api() 