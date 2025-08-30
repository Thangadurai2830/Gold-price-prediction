#!/usr/bin/env python3
"""
Debug script to test the train_models method directly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_collector import GoldDataCollector
from ml.data_processor import DataProcessor
from ml.feature_engineering import FeatureEngineer
from ml.models import GoldPricePredictor
import pandas as pd
import numpy as np

def debug_train_models():
    print("=== Testing GoldPricePredictor.train_models directly ===")
    
    # Step 1: Get processed data (same as working debug script)
    print("\n1. Getting processed data...")
    collector = GoldDataCollector()
    gold_data = collector.fetch_gold_prices_yahoo()
    
    if gold_data is None or len(gold_data) == 0:
        print("Failed to fetch real data, using sample data...")
        gold_data = collector.generate_sample_data()
    
    processor = DataProcessor()
    processor.raw_data = gold_data
    
    # Get data with technical indicators
    clean_data = processor.clean_data()
    processed_data = processor.add_technical_indicators(clean_data)
    processed_data = processed_data.dropna()
    
    print(f"Input data shape: {processed_data.shape}")
    
    # Step 2: Apply feature engineering
    print("\n2. Applying feature engineering...")
    feature_engineer = FeatureEngineer()
    
    engineered_data = feature_engineer.engineer_all_features(
        processed_data, 
        target_column='price',
        use_pca=False,
        n_features=50
    )
    
    engineered_data = engineered_data.dropna()
    print(f"Engineered data shape: {engineered_data.shape}")
    
    # Step 3: Test GoldPricePredictor.train_models
    print("\n3. Testing GoldPricePredictor.train_models...")
    
    predictor = GoldPricePredictor()
    
    try:
        # This should fail with StandardScaler error
        results = predictor.train_models(engineered_data)
        print(f"SUCCESS: Training completed with results: {results}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        # Let's see what happens in prepare_features
        print("\n4. Debugging prepare_features...")
        try:
            prepared_data = predictor.prepare_features(engineered_data, is_training=True)
            print(f"Prepared data shape: {prepared_data.shape}")
            
            if len(prepared_data) == 0:
                print("*** FOUND THE ISSUE: prepare_features returns empty data! ***")
                
                # Let's see what happens step by step
                print("\nStep-by-step debugging of prepare_features:")
                
                # Test with minimal processing
                test_data = engineered_data.copy()
                print(f"Original data shape: {test_data.shape}")
                
                # Just add one simple feature
                test_data['simple_sma_5'] = test_data['price'].rolling(window=5).mean()
                print(f"After adding SMA: {test_data.shape}")
                
                # Check NaN values
                nan_counts = test_data.isnull().sum()
                total_nans = nan_counts.sum()
                print(f"Total NaN values: {total_nans}")
                
                if total_nans > 0:
                    print("NaN values per column:")
                    for col, count in nan_counts.items():
                        if count > 0:
                            print(f"  {col}: {count}")
                
                # Drop NaN and see what's left
                test_data_clean = test_data.dropna()
                print(f"After dropna: {test_data_clean.shape}")
                
        except Exception as prep_error:
            print(f"Error in prepare_features: {prep_error}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_train_models()