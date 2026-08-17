#!/usr/bin/env python3
"""
Test the Flask app's predictor directly
"""

import sys
import os
sys.path.append('.')

import requests
import json
from flask import Flask
from app import create_app

def test_flask_predictor():
    """Test the Flask app's predictor directly"""
    print("=== FLASK PREDICTOR TEST ===")
    
    # Create the Flask app
    app = create_app()
    
    with app.app_context():
        print(f"Flask app: {app}")
        print(f"Has predictor: {hasattr(app, 'predictor')}")
        
        if hasattr(app, 'predictor'):
            predictor = app.predictor
            print(f"Predictor: {predictor}")
            print(f"Predictor type: {type(predictor)}")
            print(f"Has scaler: {hasattr(predictor, 'scaler')}")
            print(f"Has trained_models: {hasattr(predictor, 'trained_models')}")
            
            if hasattr(predictor, 'scaler'):
                print(f"Scaler type: {type(predictor.scaler)}")
                print(f"Scaler fitted: {hasattr(predictor.scaler, 'center_') and hasattr(predictor.scaler, 'scale_')}")
                
                # Test direct prediction
                try:
                    import pandas as pd
                    import numpy as np
                    
                    # Create test data
                    test_data = {
                        'economic_indicators': {
                            'inflation_rate': 3.2,
                            'interest_rate': 5.25,
                            'gdp_growth': 2.1,
                            'unemployment_rate': 3.7,
                            'dollar_index': 103.5
                        },
                        'market_data': {
                            'sp500': 4200,
                            'vix': 18.5,
                            'oil_price': 85.2,
                            'bond_yield': 4.3
                        },
                        'technical_indicators': {
                            'rsi': 65,
                            'macd': 0.8,
                            'bollinger_position': 0.7
                        },
                        'sentiment_data': {
                            'fear_greed_index': 55,
                            'news_sentiment': 0.2
                        }
                    }
                    
                    # Convert to DataFrame
                    df = pd.DataFrame([test_data])
                    print(f"Test DataFrame shape: {df.shape}")
                    print(f"Test DataFrame columns: {list(df.columns)}")
                    
                    # Try direct prediction
                    print("\n=== Testing direct prediction ===")
                    result = predictor.predict(df)
                    print(f"Prediction successful: {result}")
                    return True
                    
                except Exception as e:
                    print(f"Direct prediction failed: {e}")
                    import traceback
                    traceback.print_exc()
                    return False
            else:
                print("Predictor has no scaler")
                return False
        else:
            print("Flask app has no predictor")
            return False

if __name__ == "__main__":
    success = test_flask_predictor()
    print(f"\nFlask predictor test {'PASSED' if success else 'FAILED'}")