#!/usr/bin/env python3
"""
Test script to debug the predictor object and trained_models
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.models import GoldPricePredictor

def test_predictor():
    print("=== Testing GoldPricePredictor ===")
    
    # Create predictor instance
    predictor = GoldPricePredictor()
    print(f"Predictor created: {predictor}")
    print(f"Initial trained_models: {predictor.trained_models}")
    print(f"Initial trained_models type: {type(predictor.trained_models)}")
    print(f"Initial trained_models length: {len(predictor.trained_models)}")
    
    # Try to load models
    try:
        predictor.load_models('models')
        print(f"\nAfter loading models:")
        print(f"Trained_models: {predictor.trained_models}")
        print(f"Trained_models type: {type(predictor.trained_models)}")
        print(f"Trained_models length: {len(predictor.trained_models)}")
        print(f"Trained_models keys: {list(predictor.trained_models.keys())}")
        
        # Test the condition used in health endpoint
        models_count = len(predictor.trained_models) if predictor and hasattr(predictor, 'trained_models') else 0
        print(f"\nHealth endpoint logic test:")
        print(f"predictor is not None: {predictor is not None}")
        print(f"hasattr(predictor, 'trained_models'): {hasattr(predictor, 'trained_models')}")
        print(f"models_count: {models_count}")
        print(f"models_count > 0: {models_count > 0}")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    test_predictor()