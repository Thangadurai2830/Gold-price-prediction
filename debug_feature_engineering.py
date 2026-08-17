#!/usr/bin/env python3
"""
Debug script to identify issues in feature engineering
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_collector import GoldDataCollector
from ml.data_processor import DataProcessor
from ml.feature_engineering import FeatureEngineer
import pandas as pd
import numpy as np

def debug_feature_engineering():
    print("=== Starting Feature Engineering Debug ===")
    
    # Step 1: Get processed data
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
    print(f"Input data columns: {list(processed_data.columns)}")
    
    # Step 2: Initialize feature engineer
    print("\n2. Initializing feature engineer...")
    feature_engineer = FeatureEngineer()
    
    # Step 3: Apply feature engineering step by step
    print("\n3. Applying feature engineering...")
    
    try:
        # Apply comprehensive feature engineering
        engineered_data = feature_engineer.engineer_all_features(
            processed_data, 
            target_column='price',
            use_pca=False,
            n_features=50
        )
        
        print(f"Engineered data shape: {engineered_data.shape}")
        print(f"Selected features: {len(feature_engineer.selected_features)}")
        
        # Check for NaN values
        nan_counts = engineered_data.isnull().sum()
        total_nans = nan_counts.sum()
        print(f"\nTotal NaN values: {total_nans}")
        
        if total_nans > 0:
            print("NaN values per column:")
            for col, count in nan_counts.items():
                if count > 0:
                    print(f"  {col}: {count}")
        
        # Check final data after dropna
        final_data = engineered_data.dropna()
        print(f"\nFinal data shape after dropna: {final_data.shape}")
        
        if len(final_data) == 0:
            print("\n*** ERROR: All data was dropped after feature engineering! ***")
            
            # Try to identify the problematic features
            print("\nInvestigating problematic features...")
            
            # Check which columns have all NaN values
            all_nan_cols = engineered_data.columns[engineered_data.isnull().all()].tolist()
            if all_nan_cols:
                print(f"Columns with all NaN values: {all_nan_cols}")
            
            # Check if any row has no NaN values
            rows_without_nan = engineered_data.dropna(axis=0, how='any')
            print(f"Rows without any NaN: {len(rows_without_nan)}")
            
            return None
        
        return final_data
        
    except Exception as e:
        print(f"Error during feature engineering: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = debug_feature_engineering()
    if result is not None:
        print(f"\n=== SUCCESS: Final data shape: {result.shape} ===")
    else:
        print("\n=== FAILED: Feature engineering failed ===")