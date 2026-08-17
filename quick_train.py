#!/usr/bin/env python3
"""
Quick training script to get models working for the API
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from datetime import datetime, timedelta

def generate_sample_data(days=500):
    """Generate sample gold price data"""
    np.random.seed(42)
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate realistic gold price data
    base_price = 1800
    trend = np.linspace(0, 200, len(dates))
    seasonality = 50 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    noise = np.random.normal(0, 20, len(dates))
    
    prices = base_price + trend + seasonality + noise
    
    # Create DataFrame
    data = pd.DataFrame({
        'date': dates,
        'price': prices,
        'open': prices + np.random.normal(0, 5, len(dates)),
        'high': prices + np.random.uniform(5, 15, len(dates)),
        'low': prices - np.random.uniform(5, 15, len(dates)),
        'close': prices + np.random.normal(0, 3, len(dates)),
        'volume': np.random.randint(10000, 100000, len(dates))
    })
    
    return data

def add_features(data):
    """Add basic technical indicators"""
    df = data.copy()
    
    # Moving averages
    df['ma_5'] = df['price'].rolling(window=5).mean()
    df['ma_10'] = df['price'].rolling(window=10).mean()
    df['ma_20'] = df['price'].rolling(window=20).mean()
    
    # Price changes
    df['price_change'] = df['price'].pct_change()
    df['price_change_5'] = df['price'].pct_change(5)
    
    # Volatility
    df['volatility'] = df['price'].rolling(window=10).std()
    
    # Lag features
    df['price_lag_1'] = df['price'].shift(1)
    df['price_lag_2'] = df['price'].shift(2)
    df['price_lag_3'] = df['price'].shift(3)
    
    # Target (next day price)
    df['target'] = df['price'].shift(-1)
    
    return df

def train_model():
    """Train a simple model"""
    print("Generating sample data...")
    data = generate_sample_data()
    
    print("Adding features...")
    data_with_features = add_features(data)
    
    # Remove NaN values
    data_clean = data_with_features.dropna()
    
    # Prepare features and target
    feature_columns = ['ma_5', 'ma_10', 'ma_20', 'price_change', 'price_change_5', 
                      'volatility', 'price_lag_1', 'price_lag_2', 'price_lag_3']
    
    X = data_clean[feature_columns]
    y = data_clean['target']
    
    print(f"Training data shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"MSE: {mse:.2f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Save model
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'quick_trained_model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save feature columns
    feature_path = os.path.join(model_dir, 'feature_columns.joblib')
    joblib.dump(feature_columns, feature_path)
    print(f"Feature columns saved to {feature_path}")
    
    # Save sample data for testing
    data_path = os.path.join(model_dir, 'sample_data.csv')
    data_clean.to_csv(data_path, index=False)
    print(f"Sample data saved to {data_path}")
    
    return model, feature_columns

if __name__ == "__main__":
    print("Starting quick model training...")
    model, features = train_model()
    print("Training completed successfully!")