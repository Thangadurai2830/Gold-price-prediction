#!/usr/bin/env python3
"""
Debug script to test the exact prediction flow and identify scaler issues
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.models import GoldPricePredictor

def test_prediction_flow():
    """Test the exact prediction flow to identify scaler issues"""
    
    print("=== Testing Prediction Flow ===")
    
    # Initialize predictor
    print("1. Initializing predictor...")
    predictor = GoldPricePredictor()
    
    # Load models
    print("2. Loading models...")
    success = predictor.load_models('models')
    print(f"   Models loaded: {success}")
    print(f"   Trained models: {len(predictor.trained_models) if hasattr(predictor, 'trained_models') else 0}")
    print(f"   Feature columns: {len(predictor.feature_columns) if hasattr(predictor, 'feature_columns') else 0}")
    print(f"   Scaler type: {type(predictor.scaler) if hasattr(predictor, 'scaler') else 'None'}")
    
    # Test scaler directly
    print("\n3. Testing scaler directly...")
    try:
        test_data = np.array([[2500.0] * len(predictor.feature_columns)])
        print(f"   Test data shape: {test_data.shape}")
        result = predictor.scaler.transform(test_data)
        print(f"   ✓ Scaler transform successful, result shape: {result.shape}")
    except Exception as e:
        print(f"   ✗ Scaler transform failed: {e}")
        return
    
    # Create sample data
    print("\n4. Creating sample data...")
    sample_data = pd.DataFrame({
        'date': ['2025-08-30'],
        'price': [2500.0]
    })
    print(f"   Sample data: {sample_data}")
    
    # Test prepare_features
    print("\n5. Testing prepare_features...")
    try:
        prepared_data = predictor.prepare_features(sample_data, is_training=False)
        print(f"   ✓ Features prepared successfully")
        print(f"   Prepared data shape: {prepared_data.shape}")
        print(f"   Prepared data columns: {list(prepared_data.columns)}")
        print(f"   Feature columns expected: {predictor.feature_columns}")
    except Exception as e:
        print(f"   ✗ Feature preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test feature selection
    print("\n6. Testing feature selection...")
    try:
        X = prepared_data[predictor.feature_columns]
        print(f"   X shape: {X.shape}")
        print(f"   X columns: {list(X.columns)}")
        print(f"   X values: {X.values}")
    except Exception as e:
        print(f"   ✗ Feature selection failed: {e}")
        return
    
    # Test scaling
    print("\n7. Testing scaling...")
    try:
        X_scaled = predictor.scaler.transform(X)
        print(f"   ✓ Scaling successful")
        print(f"   Scaled shape: {X_scaled.shape}")
    except Exception as e:
        print(f"   ✗ Scaling failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test full prediction
    print("\n8. Testing full prediction...")
    try:
        prediction = predictor.predict(sample_data)
        print(f"   ✓ Prediction successful: {prediction}")
    except Exception as e:
        print(f"   ✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n=== All tests passed! ===")

if __name__ == '__main__':
    test_prediction_flow()