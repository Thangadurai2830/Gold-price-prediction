#!/usr/bin/env python3
"""
Test the Flask app's predictor directly
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
import pandas as pd

def test_app_predictor():
    """Test the Flask app's predictor"""
    print("Testing Flask app predictor...")
    
    # Initialize models
    with app.app_context():
        success = initialize_models()
        print(f"Models initialized: {success}")
        
        if hasattr(app, 'predictor'):
            predictor = app.predictor
            print(f"Predictor found: {predictor}")
            print(f"Scaler type: {type(predictor.scaler)}")
            print(f"Scaler fitted: {hasattr(predictor.scaler, 'center_') and hasattr(predictor.scaler, 'scale_')}")
            
            if hasattr(predictor.scaler, 'center_'):
                print(f"Scaler center shape: {predictor.scaler.center_.shape}")
            if hasattr(predictor.scaler, 'scale_'):
                print(f"Scaler scale shape: {predictor.scaler.scale_.shape}")
            
            print(f"Feature columns count: {len(predictor.feature_columns) if predictor.feature_columns else 0}")
            print(f"Selected features count: {len(predictor.selected_features) if predictor.selected_features is not None and not predictor.selected_features.empty else 0}")
            print(f"Best model: {predictor.best_model}")
            print(f"Trained models count: {len(predictor.trained_models) if predictor.trained_models else 0}")
            
            # Test with simple data that includes price
            test_data = pd.DataFrame([{
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
            }])
            
            print(f"\nTest data columns: {list(test_data.columns)}")
            
            try:
                result = predictor.predict(test_data)
                print(f"Prediction successful: {result}")
            except Exception as e:
                print(f"Prediction failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("No predictor found in app context")

if __name__ == "__main__":
    test_app_predictor()