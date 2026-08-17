#!/usr/bin/env python3
"""
Robust Feature Engineering Module for Gold Price Prediction

This module implements strict temporal constraints to prevent data leakage
and ensures that only past information is used to predict future prices.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class RobustFeatureEngineer:
    """
    Feature engineering with strict temporal constraints to prevent data leakage
    """
    
    def __init__(self):
        self.feature_columns = []
        self.scaler = None
        self.feature_selector = None
        
    def create_temporal_features(self, data: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """
        Create time-based features that don't cause data leakage
        """
        df = data.copy()
        
        # Ensure date column is datetime
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col).reset_index(drop=True)
            
            # Basic time features
            df['day_of_week'] = df[date_col].dt.dayofweek
            df['month'] = df[date_col].dt.month
            df['quarter'] = df[date_col].dt.quarter
            df['year'] = df[date_col].dt.year
            df['day_of_year'] = df[date_col].dt.dayofyear
            
            # Cyclical encoding for temporal features
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
            df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
            
        return df
    
    def create_lag_features(self, data: pd.DataFrame, price_col: str = 'price', 
                           lags: List[int] = None) -> pd.DataFrame:
        """
        Create lag features using only past data
        """
        if lags is None:
            lags = [1, 2, 3, 5, 7, 10, 14, 21, 30]
            
        df = data.copy()
        
        # Ensure price_col exists in the dataframe
        if price_col not in df.columns:
            raise ValueError(f"Column '{price_col}' not found in dataframe. Available columns: {list(df.columns)}")
        
        # Debug: Check for duplicate columns
        if df.columns.duplicated().any():
            print(f"Warning: Duplicate columns found: {df.columns[df.columns.duplicated()].tolist()}")
            print(f"All columns: {df.columns.tolist()}")
            # Remove duplicate columns
            df = df.loc[:, ~df.columns.duplicated()]
        
        for lag in lags:
            # Ensure we get a Series, not a DataFrame
            price_series = df[price_col]
            if isinstance(price_series, pd.DataFrame):
                # If multiple columns with same name, take the first one
                price_series = price_series.iloc[:, 0]
            df[f'{price_col}_lag_{lag}'] = price_series.shift(lag)
            
        return df
    
    def create_rolling_features(self, data: pd.DataFrame, price_col: str = 'price',
                               windows: List[int] = None) -> pd.DataFrame:
        """
        Create rolling window features with strict temporal constraints
        """
        if windows is None:
            windows = [3, 5, 7, 10, 14, 21, 30, 60]
            
        df = data.copy()
        
        for window in windows:
            # Use only past data (shift by 1 to avoid current value)
            past_data = df[price_col].shift(1)
            
            # Rolling statistics
            df[f'{price_col}_sma_{window}'] = past_data.rolling(
                window=window, min_periods=max(1, window//2)
            ).mean()
            
            df[f'{price_col}_std_{window}'] = past_data.rolling(
                window=window, min_periods=max(1, window//2)
            ).std()
            
            df[f'{price_col}_min_{window}'] = past_data.rolling(
                window=window, min_periods=max(1, window//2)
            ).min()
            
            df[f'{price_col}_max_{window}'] = past_data.rolling(
                window=window, min_periods=max(1, window//2)
            ).max()
            
            df[f'{price_col}_median_{window}'] = past_data.rolling(
                window=window, min_periods=max(1, window//2)
            ).median()
            
            # Price position within rolling window
            rolling_min = df[f'{price_col}_min_{window}']
            rolling_max = df[f'{price_col}_max_{window}']
            df[f'{price_col}_position_{window}'] = (
                past_data - rolling_min
            ) / (rolling_max - rolling_min + 1e-8)
            
        return df
    
    def create_technical_indicators(self, data: pd.DataFrame, price_col: str = 'price',
                                   high_col: str = 'high', low_col: str = 'low',
                                   volume_col: str = 'volume') -> pd.DataFrame:
        """
        Create technical indicators with temporal constraints
        """
        df = data.copy()
        
        # Ensure we have required columns
        if high_col not in df.columns:
            df[high_col] = df[price_col] * 1.01
        if low_col not in df.columns:
            df[low_col] = df[price_col] * 0.99
        if volume_col not in df.columns:
            df[volume_col] = 1000
            
        # Use shifted data to avoid leakage
        price_shifted = df[price_col].shift(1)
        high_shifted = df[high_col].shift(1)
        low_shifted = df[low_col].shift(1)
        volume_shifted = df[volume_col].shift(1)
        
        # RSI (Relative Strength Index)
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
            rs = gain / (loss + 1e-8)
            return 100 - (100 / (1 + rs))
        
        df['rsi_14'] = calculate_rsi(price_shifted, 14)
        df['rsi_30'] = calculate_rsi(price_shifted, 30)
        
        # MACD
        ema_12 = price_shifted.ewm(span=12, min_periods=1).mean()
        ema_26 = price_shifted.ewm(span=26, min_periods=1).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, min_periods=1).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        for window in [20, 50]:
            sma = price_shifted.rolling(window=window, min_periods=1).mean()
            std = price_shifted.rolling(window=window, min_periods=1).std()
            df[f'bb_upper_{window}'] = sma + (2 * std)
            df[f'bb_lower_{window}'] = sma - (2 * std)
            df[f'bb_width_{window}'] = df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']
            df[f'bb_position_{window}'] = (price_shifted - df[f'bb_lower_{window}']) / (
                df[f'bb_width_{window}'] + 1e-8
            )
        
        # Stochastic Oscillator
        def calculate_stochastic(high, low, close, k_window=14, d_window=3):
            lowest_low = low.rolling(window=k_window, min_periods=1).min()
            highest_high = high.rolling(window=k_window, min_periods=1).max()
            k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-8)
            d_percent = k_percent.rolling(window=d_window, min_periods=1).mean()
            return k_percent, d_percent
        
        df['stoch_k'], df['stoch_d'] = calculate_stochastic(
            high_shifted, low_shifted, price_shifted
        )
        
        # Williams %R
        def calculate_williams_r(high, low, close, window=14):
            highest_high = high.rolling(window=window, min_periods=1).max()
            lowest_low = low.rolling(window=window, min_periods=1).min()
            return -100 * (highest_high - close) / (highest_high - lowest_low + 1e-8)
        
        df['williams_r'] = calculate_williams_r(high_shifted, low_shifted, price_shifted)
        
        # Average True Range (ATR)
        def calculate_atr(high, low, close, window=14):
            high_low = high - low
            high_close_prev = np.abs(high - close.shift(1))
            low_close_prev = np.abs(low - close.shift(1))
            true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
            return true_range.rolling(window=window, min_periods=1).mean()
        
        df['atr'] = calculate_atr(high_shifted, low_shifted, price_shifted)
        
        return df
    
    def create_momentum_features(self, data: pd.DataFrame, price_col: str = 'price') -> pd.DataFrame:
        """
        Create momentum and change features
        """
        df = data.copy()
        
        # Use shifted data to avoid leakage
        price_shifted = df[price_col].shift(1)
        
        # Returns over different periods
        for period in [1, 2, 3, 5, 7, 10, 14, 21, 30]:
            df[f'return_{period}d'] = price_shifted.pct_change(period)
            df[f'log_return_{period}d'] = np.log(price_shifted / price_shifted.shift(period))
            
        # Momentum indicators
        for period in [5, 10, 20, 50]:
            df[f'momentum_{period}'] = price_shifted / price_shifted.shift(period) - 1
            
        # Rate of change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = (
                (price_shifted - price_shifted.shift(period)) / price_shifted.shift(period)
            ) * 100
            
        return df
    
    def create_volatility_features(self, data: pd.DataFrame, price_col: str = 'price') -> pd.DataFrame:
        """
        Create volatility-based features
        """
        df = data.copy()
        
        # Calculate returns first (using shifted data)
        price_shifted = df[price_col].shift(1)
        returns = price_shifted.pct_change()
        
        # Rolling volatility
        for window in [5, 10, 20, 30, 60]:
            df[f'volatility_{window}'] = returns.rolling(
                window=window, min_periods=max(1, window//2)
            ).std() * np.sqrt(252)  # Annualized
            
            # Realized volatility (sum of squared returns)
            df[f'realized_vol_{window}'] = (returns ** 2).rolling(
                window=window, min_periods=max(1, window//2)
            ).sum()
            
        # GARCH-like volatility clustering
        df['vol_clustering'] = returns.abs().rolling(window=10, min_periods=1).mean()
        
        return df
    
    def create_interaction_features(self, data: pd.DataFrame, max_interactions: int = 20) -> pd.DataFrame:
        """
        Create interaction features between key variables
        """
        df = data.copy()
        
        # Select key features for interactions
        key_features = []
        for col in df.columns:
            if any(x in col.lower() for x in ['sma', 'rsi', 'macd', 'return', 'volatility']):
                key_features.append(col)
        
        # Limit to prevent explosion of features
        key_features = key_features[:10]
        
        interaction_count = 0
        for i, feat1 in enumerate(key_features):
            for feat2 in key_features[i+1:]:
                if interaction_count >= max_interactions:
                    break
                    
                # Multiplicative interaction
                df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                interaction_count += 1
                
                if interaction_count >= max_interactions:
                    break
            if interaction_count >= max_interactions:
                break
        
        return df
    
    def engineer_all_features(self, data: pd.DataFrame, price_col: str = 'price',
                             date_col: str = 'date') -> pd.DataFrame:
        """
        Apply all feature engineering steps with temporal constraints
        """
        print("Starting robust feature engineering...")
        
        # Start with a copy of the data
        df = data.copy()
        
        # Ensure data is sorted by date
        if date_col in df.columns:
            df = df.sort_values(date_col).reset_index(drop=True)
        
        print(f"Initial data shape: {df.shape}")
        
        # 1. Temporal features
        print("Creating temporal features...")
        df = self.create_temporal_features(df, date_col)
        
        # 2. Lag features
        print("Creating lag features...")
        df = self.create_lag_features(df, price_col)
        
        # 3. Rolling window features
        print("Creating rolling window features...")
        df = self.create_rolling_features(df, price_col)
        
        # 4. Technical indicators
        print("Creating technical indicators...")
        df = self.create_technical_indicators(df, price_col)
        
        # 5. Momentum features
        print("Creating momentum features...")
        df = self.create_momentum_features(df, price_col)
        
        # 6. Volatility features
        print("Creating volatility features...")
        df = self.create_volatility_features(df, price_col)
        
        # 7. Interaction features (limited)
        print("Creating interaction features...")
        df = self.create_interaction_features(df, max_interactions=15)
        
        print(f"Final feature set shape: {df.shape}")
        
        # Store feature columns (excluding target and date)
        self.feature_columns = [col for col in df.columns 
                               if col not in [price_col, date_col, 'high', 'low', 'open', 'close', 'volume']]
        
        print(f"Created {len(self.feature_columns)} features")
        
        return df
    
    def prepare_training_data(self, data: pd.DataFrame, price_col: str = 'price',
                             date_col: str = 'date', min_periods: int = 60, 
                             prediction_horizon: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare training data with strict temporal validation
        Creates target variable as future price to predict
        """
        # Engineer features
        df = self.engineer_all_features(data, price_col, date_col)
        
        # Create target variable: future price (shifted backwards)
        # This ensures we're predicting future values using only past features
        df['target'] = df[price_col].shift(-prediction_horizon)
        
        # Remove rows with insufficient history and future target
        df = df.iloc[min_periods:-prediction_horizon].copy()
        
        # Separate features and target
        X = df[self.feature_columns].copy()
        y = df['target'].copy()
        
        # Remove any remaining NaN values
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        print(f"Training data shape: X={X.shape}, y={y.shape}")
        print(f"Prediction horizon: {prediction_horizon} days")
        print(f"Removed {(~valid_idx).sum()} rows with missing values")
        
        return X, y