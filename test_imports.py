#!/usr/bin/env python3
"""
Test script to check all imports and create a working predictor
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import catboost as cb
    print(f"✓ Catboost imported successfully: {cb.__version__}")
except ImportError as e:
    print(f"✗ Catboost import failed: {e}")

try:
    from ml.models import GoldPricePredictor
    print("✓ GoldPricePredictor imported successfully")
    
    # Test creating an instance
    predictor = GoldPricePredictor()
    print("✓ GoldPricePredictor instance created")
    
    # Test loading models
    result = predictor.load_models('models')
    print(f"✓ Models loaded: {result}")
    print(f"✓ Trained models: {list(predictor.trained_models.keys()) if hasattr(predictor, 'trained_models') else 'No trained_models'}")
    print(f"✓ Best model: {predictor.best_model is not None if hasattr(predictor, 'best_model') else 'No best_model'}")
    
except ImportError as e:
    print(f"✗ GoldPricePredictor import failed: {e}")
except Exception as e:
    print(f"✗ Error creating/testing predictor: {e}")
    import traceback
    traceback.print_exc()