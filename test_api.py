import requests
import json

def test_api_endpoints():
    """Test all API endpoints with proper error handling"""
    base_url = "http://localhost:5000/api"
    
    print("Testing API endpoints with comprehensive error handling...")
    
    # Test 1: Ping endpoint
    try:
        print("\n1. Testing /api/ping endpoint:")
        response = requests.get(f"{base_url}/ping")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:200]}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: Health endpoint
    try:
        print("\n2. Testing /api/health endpoint:")
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:200]}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 3: Data latest endpoint
    try:
        print("\n3. Testing /api/data/latest endpoint:")
        response = requests.get(f"{base_url}/data/latest")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response keys: {list(data.keys())}")
            print(f"Data source: {data.get('data_source', 'unknown')}")
            print(f"Latest price: {data.get('latest_price', 'N/A')}")
        else:
            print(f"Response: {response.text[:300]}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 4: Predict single endpoint
    try:
        print("\n4. Testing /api/predict/single endpoint:")
        test_data = {
            "features": {
                "price_lag_1": 2000.0,
                "sma_5": 2000.0,
                "volatility": 0.02,
                "price_change": 0.0,
                "volume": 100000
            }
        }
        response = requests.post(f"{base_url}/predict/single", json=test_data)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response keys: {list(data.keys())}")
            print(f"Prediction: {data.get('prediction', data.get('fallback_prediction', 'N/A'))}")
            print(f"Model used: {data.get('model_used', data.get('model_type', 'unknown'))}")
        else:
            print(f"Response: {response.text[:300]}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 5: Analysis summary endpoint
    try:
        print("\n5. Testing /api/analysis/summary endpoint:")
        response = requests.get(f"{base_url}/analysis/summary")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response keys: {list(data.keys())}")
            if 'price_statistics' in data:
                print(f"Current price: {data['price_statistics'].get('current', 'N/A')}")
                print(f"Data source: {data.get('data_source', 'unknown')}")
            elif 'fallback_analysis' in data:
                print(f"Fallback analysis: {data['fallback_analysis']}")
        else:
            print(f"Response: {response.text[:300]}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 6: Predict ensemble endpoint
    try:
        print("\n6. Testing /api/predict/ensemble endpoint:")
        response = requests.post(f"{base_url}/predict/ensemble", json={})
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response keys: {list(data.keys())}")
            print(f"Ensemble prediction: {data.get('ensemble_prediction', 'N/A')}")
            print(f"Successful models: {data.get('successful_models', 0)}")
        else:
            print(f"Response: {response.text[:300]}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\nAPI endpoint testing completed!")
    print("All endpoints should return status 200 with proper fallback responses.")

if __name__ == "__main__":
    test_api_endpoints()