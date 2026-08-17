#!/usr/bin/env python3
"""
Simple test to verify Flask app and predictor
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app

def test_flask_app():
    print("=== Testing Flask App ===")
    
    app = create_app()
    
    with app.app_context():
        print(f"App created: {app}")
        print(f"App has predictor: {hasattr(app, 'predictor')}")
        
        if hasattr(app, 'predictor'):
            predictor = app.predictor
            print(f"Predictor: {predictor}")
            print(f"Predictor type: {type(predictor)}")
            print(f"Has trained_models: {hasattr(predictor, 'trained_models')}")
            
            if hasattr(predictor, 'trained_models'):
                print(f"Trained models: {predictor.trained_models}")
                print(f"Models count: {len(predictor.trained_models)}")
                print(f"Model keys: {list(predictor.trained_models.keys())}")
                
                # Test the health endpoint logic
                models_count = len(predictor.trained_models)
                models_loaded = models_count > 0
                print(f"\nHealth endpoint logic:")
                print(f"models_count: {models_count}")
                print(f"models_loaded: {models_loaded}")
        
        # Test the get_predictor function
        from app.api import get_predictor
        api_predictor = get_predictor()
        print(f"\nAPI get_predictor result: {api_predictor}")
        if api_predictor and hasattr(api_predictor, 'trained_models'):
            print(f"API predictor models count: {len(api_predictor.trained_models)}")

if __name__ == "__main__":
    test_flask_app()