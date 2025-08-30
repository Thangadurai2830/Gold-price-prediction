import pandas as pd
import numpy as np
from ml.models import GoldPricePredictor
import joblib

# Test the scaler issue
predictor = GoldPricePredictor()

# Load models to get the scaler
predictor.load_models('models')

print(f"Scaler type: {type(predictor.scaler)}")
print(f"Scaler fitted: {hasattr(predictor.scaler, 'center_') and hasattr(predictor.scaler, 'scale_')}")
print(f"Feature columns length: {len(predictor.feature_columns)}")
print(f"Selected features length: {len(predictor.selected_features)}")

# Create test data
test_data = pd.DataFrame({
    'price': [2300.0],
    'date': ['2024-01-01']
})

print("\nTesting prepare_features...")
try:
    features = predictor.prepare_features(test_data, is_training=False)
    print(f"Features generated: {features.shape}")
    print(f"Feature columns: {list(features.columns)}")
    
    # Test if we can create the expected feature matrix
    X = pd.DataFrame(index=features.index)
    for feature in predictor.feature_columns:
        if feature in features.columns:
            X[feature] = features[feature]
        else:
            X[feature] = 0.0  # Default value
    
    print(f"\nX shape: {X.shape}")
    print(f"Expected features: {len(predictor.feature_columns)}")
    
    # Test scaler transform
    print("\nTesting scaler transform...")
    X_scaled = predictor.scaler.transform(X)
    print(f"Transform successful! Shape: {X_scaled.shape}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()