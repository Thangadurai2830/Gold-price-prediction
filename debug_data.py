#!/usr/bin/env python3
"""
Debug script to identify where data is being lost during preprocessing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_collector import GoldDataCollector
from ml.data_processor import DataProcessor
from ml.feature_engineering import FeatureEngineer
import pandas as pd

def debug_data_processing():
    print("=== Starting Data Debug Process ===")
    
    # Step 1: Collect data
    print("\n1. Collecting data...")
    collector = GoldDataCollector()
    gold_data = collector.fetch_gold_prices_yahoo()
    
    if gold_data is None or len(gold_data) == 0:
        print("Failed to fetch real data, using sample data...")
        gold_data = collector.generate_sample_data()
    
    print(f"Raw gold data shape: {gold_data.shape}")
    print(f"Raw gold data columns: {list(gold_data.columns)}")
    print(f"Raw gold data head:\n{gold_data.head()}")
    
    # Step 2: Initialize processor
    print("\n2. Initializing data processor...")
    processor = DataProcessor()
    processor.raw_data = gold_data
    
    # Step 3: Clean data
    print("\n3. Cleaning data...")
    clean_data = processor.clean_data()
    print(f"Clean data shape: {clean_data.shape}")
    print(f"Clean data columns: {list(clean_data.columns)}")
    
    # Step 4: Add technical indicators
    print("\n4. Adding technical indicators...")
    data_with_indicators = processor.add_technical_indicators(clean_data)
    print(f"Data with indicators shape: {data_with_indicators.shape}")
    print(f"Data with indicators columns: {list(data_with_indicators.columns)}")
    
    # Check for NaN values before dropna
    print(f"\nNaN values per column before dropna:")
    nan_counts = data_with_indicators.isnull().sum()
    for col, count in nan_counts.items():
        if count > 0:
            print(f"  {col}: {count}")
    
    # Step 5: Drop NaN values
    print("\n5. Dropping NaN values...")
    final_data = data_with_indicators.dropna()
    print(f"Final data shape after dropna: {final_data.shape}")
    
    if len(final_data) == 0:
        print("\n*** ERROR: All data was dropped! ***")
        print("Investigating...")
        
        # Check which columns have all NaN values
        all_nan_cols = data_with_indicators.columns[data_with_indicators.isnull().all()].tolist()
        if all_nan_cols:
            print(f"Columns with all NaN values: {all_nan_cols}")
        
        # Check if any row has no NaN values
        rows_without_nan = data_with_indicators.dropna(axis=0, how='any')
        print(f"Rows without any NaN: {len(rows_without_nan)}")
        
        # Try dropping only columns that are all NaN
        data_drop_cols = data_with_indicators.dropna(axis=1, how='all')
        print(f"Data after dropping all-NaN columns: {data_drop_cols.shape}")
        
        # Try forward fill before dropna
        data_ffill = data_with_indicators.fillna(method='ffill')
        data_ffill_dropna = data_ffill.dropna()
        print(f"Data after forward fill and dropna: {data_ffill_dropna.shape}")
        
        return None
    
    return final_data

if __name__ == "__main__":
    result = debug_data_processing()
    if result is not None:
        print(f"\n=== SUCCESS: Final data shape: {result.shape} ===")
    else:
        print("\n=== FAILED: No data remaining after processing ===")