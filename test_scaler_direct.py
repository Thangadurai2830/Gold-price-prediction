#!/usr/bin/env python3
"""
Direct scaler test to verify the scaler is working correctly
"""

import sys
import os
sys.path.append('.')

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

def test_scaler_directly():
    """Test the scaler directly from the saved file"""
    print("=== DIRECT SCALER TEST ===")
    
    # Load the scaler directly
    scaler_path = 'models/scaler.pkl'
    if os.path.exists(scaler_path):
        print(f"Loading scaler from {scaler_path}")
        scaler = joblib.load(scaler_path)
        print(f"Scaler type: {type(scaler)}")
        print(f"Scaler fitted: {hasattr(scaler, 'center_') and hasattr(scaler, 'scale_')}")
        
        if hasattr(scaler, 'center_'):
            print(f"Scaler center shape: {scaler.center_.shape}")
            print(f"Scaler scale shape: {scaler.scale_.shape}")
            print(f"Scaler center (first 10): {scaler.center_[:10]}")
            print(f"Scaler scale (first 10): {scaler.scale_[:10]}")
        
        # Load feature columns
        feature_columns_path = 'models/feature_columns.pkl'
        if os.path.exists(feature_columns_path):
            feature_columns = joblib.load(feature_columns_path)
            print(f"Feature columns count: {len(feature_columns)}")
            print(f"Feature columns (first 10): {feature_columns[:10]}")
            
            # Test with sample data matching the feature count
            print("\n=== Testing scaler transform ===")
            test_data = np.random.normal(2000, 100, (1, len(feature_columns)))
            print(f"Test data shape: {test_data.shape}")
            print(f"Test data: {test_data[0][:10]}")
            
            try:
                transformed = scaler.transform(test_data)
                print(f"Transform successful!")
                print(f"Transformed shape: {transformed.shape}")
                print(f"Transformed data: {transformed[0][:10]}")
                return True
            except Exception as e:
                print(f"Transform failed: {e}")
                return False
        else:
            print(f"Feature columns file not found: {feature_columns_path}")
            return False
    else:
        print(f"Scaler file not found: {scaler_path}")
        return False

if __name__ == "__main__":
    success = test_scaler_directly()
    print(f"\nScaler test {'PASSED' if success else 'FAILED'}")