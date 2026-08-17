#!/usr/bin/env python3
"""
Test the Flask API directly using test client
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from the app.py file directly
import importlib.util
spec = importlib.util.spec_from_file_location("app_module", "app.py")
app_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_module)
app = app_module.app
initialize_models = app_module.initialize_models

def test_flask_api_direct():
    """Test the Flask API using test client"""
    print("Testing Flask API directly...")
    
    # Initialize models
    with app.app_context():
        success = initialize_models()
        print(f"Models initialized: {success}")
        
        # Create test client
        client = app.test_client()
        
        # Test data
        test_data = {
            'data': {
                'price': 2000.0,
                'inflation_rate': 3.2,
                'interest_rate': 5.25,
                'gdp_growth': 2.1,
                'unemployment_rate': 3.7,
                'dollar_index': 103.5,
                'sp500': 4200,
                'vix': 18.5,
                'oil_price': 85.2,
                'bond_yield': 4.3,
                'rsi': 65,
                'macd': 0.8,
                'bollinger_position': 0.7,
                'fear_greed_index': 55,
                'news_sentiment': 0.2
            }
        }
        
        print(f"\nTesting /api/predict/advanced endpoint...")
        
        # Test the endpoint
        response = client.post('/api/predict/advanced', 
                             json=test_data,
                             content_type='application/json')
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.get_json()}")
        
        if response.status_code != 200:
            print(f"Error response: {response.data.decode()}")

if __name__ == "__main__":
    test_flask_api_direct()