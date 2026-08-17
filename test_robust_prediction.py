#!/usr/bin/env python3
"""
Test script for robust prediction with proper feature engineering
"""

import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib

def create_sample_data():
    """
    Create sample gold price data that matches the robust feature engineering pipeline
    """
    # Create sample time series data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    
    # Generate realistic gold price data
    base_price = 2000
    price_changes = np.random.normal(0, 20, len(dates))
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] + change
        prices.append(max(1500, min(2500, new_price)))  # Keep within realistic bounds
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'price': prices,
        'volume': np.random.normal(1000, 200, len(dates))
    })
    
    return df

def engineer_features(data):
    """
    Engineer features using the robust pipeline
    """
    # Load robust components
    try:
        scaler = joblib.load('models/robust_scaler_20250830_181132.pkl')
        selector = joblib.load('models/robust_selector_20250830_181132.pkl')
        feature_names = joblib.load('models/robust_features_20250830_181132.pkl')
        print(f"Loaded {len(feature_names)} feature names")
    except Exception as e:
        print(f"Error loading robust components: {e}")
        return None
    
    # Create a DataFrame with the expected features
    latest_data = pd.DataFrame([data])
    
    # Add missing features with default values
    for feature in feature_names:
        if feature not in latest_data.columns:
            if 'price' in feature.lower() or 'close' in feature.lower():
                latest_data[feature] = 2000.0  # Default gold price
            elif 'volume' in feature.lower():
                latest_data[feature] = 1000000.0  # Default volume
            elif 'return' in feature.lower() or 'change' in feature.lower():
                latest_data[feature] = 0.01  # Default return
            elif 'volatility' in feature.lower() or 'std' in feature.lower():
                latest_data[feature] = 0.02  # Default volatility
            elif 'ma' in feature.lower() or 'sma' in feature.lower() or 'ema' in feature.lower():
                latest_data[feature] = 2000.0  # Default moving average
            elif 'rsi' in feature.lower():
                latest_data[feature] = 50.0  # Default RSI
            elif 'bb' in feature.lower() or 'bollinger' in feature.lower():
                latest_data[feature] = 0.5  # Default Bollinger Band
            elif 'macd' in feature.lower():
                latest_data[feature] = 0.0  # Default MACD
            elif 'atr' in feature.lower():
                latest_data[feature] = 20.0  # Default ATR
            elif 'adx' in feature.lower():
                latest_data[feature] = 25.0  # Default ADX
            elif 'cci' in feature.lower():
                latest_data[feature] = 0.0  # Default CCI
            elif 'williams' in feature.lower():
                latest_data[feature] = -50.0  # Default Williams %R
            elif 'stoch' in feature.lower():
                latest_data[feature] = 50.0  # Default Stochastic
            elif 'momentum' in feature.lower():
                latest_data[feature] = 0.0  # Default Momentum
            elif 'roc' in feature.lower():
                latest_data[feature] = 0.01  # Default ROC
            elif 'obv' in feature.lower():
                latest_data[feature] = 1000000.0  # Default OBV
            elif 'vwap' in feature.lower():
                latest_data[feature] = 2000.0  # Default VWAP
            elif 'lag' in feature.lower():
                latest_data[feature] = 2000.0  # Default lag feature
            elif 'rolling' in feature.lower():
                latest_data[feature] = 2000.0  # Default rolling feature
            elif 'trend' in feature.lower():
                latest_data[feature] = 1.0  # Default trend
            elif 'seasonal' in feature.lower():
                latest_data[feature] = 0.0  # Default seasonal
            elif 'cycle' in feature.lower():
                latest_data[feature] = 0.0  # Default cycle
            elif 'fourier' in feature.lower():
                latest_data[feature] = 0.0  # Default Fourier
            elif 'wavelet' in feature.lower():
                latest_data[feature] = 0.0  # Default wavelet
            elif 'pca' in feature.lower():
                latest_data[feature] = 0.0  # Default PCA
            elif 'ica' in feature.lower():
                latest_data[feature] = 0.0  # Default ICA
            elif 'cluster' in feature.lower():
                latest_data[feature] = 0  # Default cluster
            elif 'regime' in feature.lower():
                latest_data[feature] = 0  # Default regime
            elif 'anomaly' in feature.lower():
                latest_data[feature] = 0.0  # Default anomaly score
            elif 'entropy' in feature.lower():
                latest_data[feature] = 0.5  # Default entropy
            elif 'fractal' in feature.lower():
                latest_data[feature] = 1.5  # Default fractal dimension
            elif 'hurst' in feature.lower():
                latest_data[feature] = 0.5  # Default Hurst exponent
            elif 'lyapunov' in feature.lower():
                latest_data[feature] = 0.0  # Default Lyapunov exponent
            elif 'correlation' in feature.lower():
                latest_data[feature] = 0.0  # Default correlation
            elif 'cointegration' in feature.lower():
                latest_data[feature] = 0.0  # Default cointegration
            elif 'granger' in feature.lower():
                latest_data[feature] = 0.0  # Default Granger causality
            elif 'var' in feature.lower() and 'model' in feature.lower():
                latest_data[feature] = 0.0  # Default VAR model
            elif 'garch' in feature.lower():
                latest_data[feature] = 0.02  # Default GARCH
            elif 'arch' in feature.lower():
                latest_data[feature] = 0.02  # Default ARCH
            elif 'egarch' in feature.lower():
                latest_data[feature] = 0.02  # Default EGARCH
            elif 'gjr' in feature.lower():
                latest_data[feature] = 0.02  # Default GJR-GARCH
            elif 'tgarch' in feature.lower():
                latest_data[feature] = 0.02  # Default TGARCH
            elif 'figarch' in feature.lower():
                latest_data[feature] = 0.02  # Default FIGARCH
            elif 'aparch' in feature.lower():
                latest_data[feature] = 0.02  # Default APARCH
            elif 'news' in feature.lower() or 'sentiment' in feature.lower():
                latest_data[feature] = 0.0  # Default sentiment
            elif 'economic' in feature.lower() or 'macro' in feature.lower():
                latest_data[feature] = 0.0  # Default economic indicator
            elif 'geopolitical' in feature.lower():
                latest_data[feature] = 0.0  # Default geopolitical risk
            elif 'weather' in feature.lower():
                latest_data[feature] = 0.0  # Default weather impact
            elif 'supply' in feature.lower() or 'demand' in feature.lower():
                latest_data[feature] = 0.0  # Default supply/demand
            elif 'mining' in feature.lower():
                latest_data[feature] = 0.0  # Default mining data
            elif 'central_bank' in feature.lower():
                latest_data[feature] = 0.0  # Default central bank data
            elif 'inflation' in feature.lower():
                latest_data[feature] = 0.02  # Default inflation
            elif 'interest_rate' in feature.lower():
                latest_data[feature] = 0.05  # Default interest rate
            elif 'gdp' in feature.lower():
                latest_data[feature] = 0.02  # Default GDP growth
            elif 'unemployment' in feature.lower():
                latest_data[feature] = 0.05  # Default unemployment
            elif 'currency' in feature.lower() or 'exchange' in feature.lower():
                latest_data[feature] = 1.0  # Default exchange rate
            elif 'oil' in feature.lower() or 'crude' in feature.lower():
                latest_data[feature] = 70.0  # Default oil price
            elif 'silver' in feature.lower():
                latest_data[feature] = 25.0  # Default silver price
            elif 'platinum' in feature.lower():
                latest_data[feature] = 1000.0  # Default platinum price
            elif 'palladium' in feature.lower():
                latest_data[feature] = 2000.0  # Default palladium price
            elif 'copper' in feature.lower():
                latest_data[feature] = 4.0  # Default copper price
            elif 'bond' in feature.lower() or 'yield' in feature.lower():
                latest_data[feature] = 0.03  # Default bond yield
            elif 'stock' in feature.lower() or 'equity' in feature.lower():
                latest_data[feature] = 4000.0  # Default stock index
            elif 'vix' in feature.lower():
                latest_data[feature] = 20.0  # Default VIX
            elif 'dollar' in feature.lower() or 'dxy' in feature.lower():
                latest_data[feature] = 100.0  # Default dollar index
            else:
                latest_data[feature] = 0.0  # Default fallback
    
    # Extract features in the correct order
    X = latest_data[feature_names].values
    
    print(f"Feature vector shape: {X.shape}")
    print(f"Sample features: {X[0][:10]}")
    
    # Apply scaling first (expects 138 features)
    X_scaled = scaler.transform(X)
    print(f"Scaled features shape: {X_scaled.shape}")
    
    # Then apply feature selection (reduces to 50 features)
    X_selected = selector.transform(X_scaled)
    print(f"Selected features shape: {X_selected.shape}")
    
    return X_selected[0].tolist()  # Return as list for JSON serialization

def test_prediction():
    """
    Test the robust prediction pipeline
    """
    print("Creating sample data...")
    
    # Create sample input data
    sample_data = {
        'price': 2050.0,
        'volume': 1200000.0,
        'returns': 0.015,
        'volatility': 0.025
    }
    
    print("Engineering features...")
    features = engineer_features(sample_data)
    
    if features is None:
        print("Failed to engineer features")
        return
    
    print(f"Testing prediction with {len(features)} features...")
    
    # Test the prediction endpoint with Lasso model
    data = {
        'features': features,
        'model': 'lasso'  # Use the Lasso model that was trained with 138 features
    }
    
    try:
        response = requests.post('http://localhost:5000/api/predict/single', json=data)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            result = response.json()
            if 'error' not in result:
                print("✓ Prediction successful!")
                print(f"Predicted price: ${result.get('prediction', 'N/A')}")
                print(f"Model used: {result.get('model_used', 'N/A')}")
            else:
                print(f"✗ Prediction failed: {result['error']}")
        else:
            print(f"✗ HTTP error: {response.status_code}")
            
    except Exception as e:
        print(f"✗ Request failed: {e}")

if __name__ == "__main__":
    test_prediction()