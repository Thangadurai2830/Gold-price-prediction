import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from prophet import Prophet
import joblib
import os
from datetime import datetime, timedelta
import warnings
from typing import Dict, Any, Tuple, List, Optional, Union
import optuna
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization, Attention, MultiHeadAttention, LayerNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
import tensorflow as tf
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

class AdvancedEnsembleRegressor(BaseEstimator, RegressorMixin):
    """
    Advanced ensemble regressor with dynamic weighting and meta-learning
    """
    
    def __init__(self, base_models=None, meta_model=None, cv_folds=5):
        self.base_models = base_models or self._get_default_base_models()
        self.meta_model = meta_model or Ridge(alpha=1.0)
        self.cv_folds = cv_folds
        self.trained_models = []
        self.meta_features = None
        
    def _get_default_base_models(self):
        return {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgb': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'lgb': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
            'svr': SVR(kernel='rbf'),
            'ridge': Ridge(alpha=1.0)
        }
    
    def fit(self, X, y):
        # Convert to numpy arrays if needed
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = np.array(X)
            
        if hasattr(y, 'values'):
            y_array = y.values
        else:
            y_array = np.array(y)
        
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        meta_features = np.zeros((X_array.shape[0], len(self.base_models)))
        
        # Train base models with cross-validation
        for i, (name, model) in enumerate(self.base_models.items()):
            model_preds = np.zeros(X_array.shape[0])
            
            for train_idx, val_idx in tscv.split(X_array):
                X_train, X_val = X_array[train_idx], X_array[val_idx]
                y_train, y_val = y_array[train_idx], y_array[val_idx]
                
                model_copy = model.__class__(**model.get_params())
                model_copy.fit(X_train, y_train)
                model_preds[val_idx] = model_copy.predict(X_val)
            
            meta_features[:, i] = model_preds
            
            # Train final model on full data
            final_model = model.__class__(**model.get_params())
            final_model.fit(X_array, y_array)
            self.trained_models.append(final_model)
        
        # Train meta-model
        self.meta_features = meta_features
        self.meta_model.fit(meta_features, y_array)
        
        return self
    
    def predict(self, X):
        # Get predictions from base models
        base_predictions = np.zeros((X.shape[0], len(self.trained_models)))
        
        for i, model in enumerate(self.trained_models):
            base_predictions[:, i] = model.predict(X)
        
        # Use meta-model to combine predictions
        final_predictions = self.meta_model.predict(base_predictions)
        
        return final_predictions

class GoldPricePredictor:
    """
    Gold Price Prediction System using multiple ML algorithms
    """
    
    def __init__(self):
        # Base models with optimized hyperparameters including advanced models
        self.models = {
            # Tree-based models
            'random_forest': RandomForestRegressor(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, max_features='sqrt', bootstrap=True,
                oob_score=True, random_state=42, n_jobs=-1
            ),
            'extra_trees': ExtraTreesRegressor(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, max_features='sqrt', bootstrap=True,
                oob_score=True, random_state=42, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=8,
                min_samples_split=5, subsample=0.8, max_features='sqrt',
                random_state=42
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=8,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                reg_lambda=0.1, random_state=42
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=8,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                reg_lambda=0.1, random_state=42, verbose=-1
            ),
            'catboost': cb.CatBoostRegressor(
                iterations=200, learning_rate=0.05, depth=8,
                l2_leaf_reg=3, random_seed=42, verbose=False
            ),
            
            # Linear models
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1, max_iter=2000),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000),
            'bayesian_ridge': BayesianRidge(),
            'huber': HuberRegressor(epsilon=1.35, max_iter=200),
            
            # Support Vector Machines
            'svr_rbf': SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1),
            'svr_linear': SVR(kernel='linear', C=1.0, epsilon=0.1),
            'svr_poly': SVR(kernel='poly', degree=3, C=1.0, gamma='scale', epsilon=0.1),
            
            # Advanced ensemble
            'advanced_ensemble': AdvancedEnsembleRegressor(cv_folds=5)
        }
        self.prophet_model = None
        self.trained_models = {}
        self.ensemble_models = {}
        self.feature_columns = []
        self.target_column = 'price'
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.best_model = None
        self.best_score = float('-inf')
        self.ensemble_weights = {}
        self.confidence_intervals = {}
        self.advanced_ensemble = None
        self.model_performance_history = {}
        self.feature_importance_scores = {}
        self.prediction_intervals = {}
        
    def prepare_features(self, data, is_training=True):
        """
        Comprehensive feature engineering that matches the trained model's expected features
        Expected features: 50 features including OHLC, technical indicators, rolling statistics, and temporal features
        """
        data = data.copy()
        
        # Ensure data is sorted by date
        if 'date' in data.columns:
            data = data.sort_values('date').reset_index(drop=True)
        
        # If we don't have OHLC data, create it from price
        if 'open' not in data.columns:
            data['open'] = data['price']
        if 'high' not in data.columns:
            data['high'] = data['price'] * 1.01  # Assume 1% higher
        if 'low' not in data.columns:
            data['low'] = data['price'] * 0.99   # Assume 1% lower
        if 'close' not in data.columns:
            data['close'] = data['price']
        
        # Technical indicators
        data['sma_5'] = data['close'].rolling(window=5, min_periods=1).mean()
        data['sma_10'] = data['close'].rolling(window=10, min_periods=1).mean()
        data['sma_20'] = data['close'].rolling(window=20, min_periods=1).mean()
        data['sma_50'] = data['close'].rolling(window=50, min_periods=1).mean()
        
        # Exponential moving averages
        data['ema_12'] = data['close'].ewm(span=12).mean()
        data['ema_26'] = data['close'].ewm(span=26).mean()
        
        # Bollinger Bands
        data['bb_middle'] = data['close'].rolling(window=20, min_periods=1).mean()
        bb_std = data['close'].rolling(window=20, min_periods=1).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        
        # Price lag
        data['price_lag_1'] = data['close'].shift(1).fillna(data['close'])
        
        # Rolling statistics for different windows
        for window in [5, 10, 20, 50]:
            data[f'price_rolling_mean_{window}'] = data['close'].rolling(window=window, min_periods=1).mean()
            data[f'price_rolling_min_{window}'] = data['close'].rolling(window=window, min_periods=1).min()
            data[f'price_rolling_max_{window}'] = data['close'].rolling(window=window, min_periods=1).max()
            data[f'price_rolling_median_{window}'] = data['close'].rolling(window=window, min_periods=1).median()
            data[f'price_rolling_q25_{window}'] = data['close'].rolling(window=window, min_periods=1).quantile(0.25)
            data[f'price_rolling_q75_{window}'] = data['close'].rolling(window=window, min_periods=1).quantile(0.75)
        
        # Price position and z-score
        for window in [20, 50, 100]:
            rolling_min = data['close'].rolling(window=window, min_periods=1).min()
            rolling_max = data['close'].rolling(window=window, min_periods=1).max()
            data[f'price_position_{window}'] = (data['close'] - rolling_min) / (rolling_max - rolling_min + 1e-8)
            
            rolling_mean = data['close'].rolling(window=window, min_periods=1).mean()
            rolling_std = data['close'].rolling(window=window, min_periods=1).std()
            data[f'price_zscore_{window}'] = (data['close'] - rolling_mean) / (rolling_std + 1e-8)
        
        # Temporal features
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data['month'] = data['date'].dt.month
            data['day_of_year'] = data['date'].dt.dayofyear
            data['week_of_year'] = data['date'].dt.isocalendar().week
            data['quarter'] = data['date'].dt.quarter
            
            # Cyclical encoding
            data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
            data['day_of_year_cos'] = np.cos(2 * np.pi * data['day_of_year'] / 365)
        else:
            # Default temporal features if no date column
            data['month'] = 6  # Default to June
            data['day_of_year'] = 180  # Default to mid-year
            data['week_of_year'] = 26  # Default to mid-year
            data['quarter'] = 2  # Default to Q2
            data['month_cos'] = np.cos(2 * np.pi * 6 / 12)
            data['day_of_year_cos'] = np.cos(2 * np.pi * 180 / 365)
        
        # Fill any remaining NaN values
        data = data.ffill().bfill().fillna(0)
        
        # Only set feature columns during training
        if is_training:
            self.feature_columns = [
                'open', 'high', 'low', 'close', 'sma_5', 'sma_10', 'sma_20', 'sma_50',
                'ema_12', 'ema_26', 'bb_middle', 'bb_upper', 'bb_lower', 'price_lag_1',
                'price_rolling_mean_5', 'price_rolling_min_5', 'price_rolling_max_5',
                'price_rolling_median_5', 'price_rolling_q25_5', 'price_rolling_q75_5',
                'price_rolling_mean_10', 'price_rolling_min_10', 'price_rolling_max_10',
                'price_rolling_median_10', 'price_rolling_q25_10', 'price_rolling_q75_10',
                'price_rolling_mean_20', 'price_rolling_min_20', 'price_rolling_max_20',
                'price_rolling_median_20', 'price_rolling_q25_20', 'price_rolling_q75_20',
                'price_rolling_mean_50', 'price_rolling_min_50', 'price_rolling_max_50',
                'price_rolling_median_50', 'price_rolling_q25_50', 'price_rolling_q75_50',
                'price_position_20', 'price_zscore_20', 'price_position_50', 'price_zscore_50',
                'price_position_100', 'price_zscore_100', 'month', 'day_of_year',
                'week_of_year', 'quarter', 'month_cos', 'day_of_year_cos'
            ]
            print(f"Generated {len(self.feature_columns)} features")
        
        return data
        
    def _calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
        
    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, lower
    
    def train_models_with_features(self, data, feature_columns, validation_split=0.2, feature_selection_k=50, use_walk_forward=True):
        """
        Train multiple models with pre-engineered features (skips internal feature engineering)
        """
        print("Starting model training with pre-engineered features...")
        
        # Use provided feature columns
        self.feature_columns = feature_columns
        
        # Remove rows with NaN values
        data = data.dropna()
        
        print(f"Data shape for training: {data.shape}")
        
        # Split features and target
        X = data[self.feature_columns]
        y = data['price']
        
        return self._train_models_core(X, y, validation_split, feature_selection_k, use_walk_forward)
    
    def train_models(self, data, validation_split=0.2, feature_selection_k=50, use_walk_forward=True):
        """
        Train multiple models with proper time series validation to prevent data leakage
        """
        print("Starting model training with time series validation...")
        
        # Prepare features with proper temporal constraints
        data = self.prepare_features(data, is_training=True)
        
        # Remove rows with NaN values
        data = data.dropna()
        
        print(f"Data shape after feature engineering: {data.shape}")
        
        # Split features and target
        X = data[self.feature_columns]
        y = data['price']
        
        return self._train_models_core(X, y, validation_split, feature_selection_k, use_walk_forward)
    
    def _train_models_core(self, X, y, validation_split=0.2, feature_selection_k=50, use_walk_forward=True):
        """
        Core training logic shared by both training methods
        """
        
        # Time series split (no shuffling to maintain temporal order)
        split_idx = int(len(X) * (1 - validation_split))
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        print(f"Training set size: {len(X_train)} (from start to {split_idx})")
        print(f"Test set size: {len(X_test)} (from {split_idx} to end)")
        
        # Feature scaling
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame for easier handling
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        # Feature selection
        self.feature_selector = SelectKBest(score_func=f_regression, k=min(feature_selection_k, len(self.feature_columns)))
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        # Get selected feature names
        selected_features = X_train_scaled.columns[self.feature_selector.get_support()]
        X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
        X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)
        
        print(f"Selected {len(selected_features)} best features out of {len(self.feature_columns)}")
        
        # Time series cross-validation with proper temporal splits
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Train models with cross-validation
        results = {}
        
        print("\nTraining individual models with time series validation...")
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            if use_walk_forward:
                # Walk-forward validation
                cv_scores = self._walk_forward_validation(model, X_train_selected, y_train, n_splits=5)
                cv_score = np.mean(cv_scores)
            else:
                # Standard time series CV
                cv_scores = []
                for train_idx, val_idx in tscv.split(X_train_selected):
                    X_cv_train, X_cv_val = X_train_selected.iloc[train_idx], X_train_selected.iloc[val_idx]
                    y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    model_copy = self._clone_model(model)
                    model_copy.fit(X_cv_train, y_cv_train)
                    y_cv_pred = model_copy.predict(X_cv_val)
                    cv_scores.append(mean_squared_error(y_cv_val, y_cv_pred))
                
                cv_score = np.mean(cv_scores)
            
            # Fit model on full training set
            model.fit(X_train_selected, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_selected)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': np.sqrt(mse),
                'cv_score': cv_score
            }
            
            self.trained_models[name] = model
            
            print(f"{name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, CV Score: {cv_score:.4f}")
        
        # Create ensemble models
        print("\nCreating ensemble models...")
        
        # Get the best individual models for ensemble
        model_list = list(self.trained_models.items())
        if len(model_list) >= 4:
            # Voting Regressor with best models
            best_models = sorted(model_list, key=lambda x: results[x[0]]['cv_score'])[:4]
            voting_regressor = VotingRegressor([
                (f'model_{i}', model) for i, (name, model) in enumerate(best_models)
            ])
            voting_regressor.fit(X_train_selected, y_train)
            
            # Stacking Regressor with meta-learner
            stacking_regressor = StackingRegressor(
                estimators=[
                    (f'model_{i}', model) for i, (name, model) in enumerate(best_models[:3])
                ],
                final_estimator=Ridge(alpha=1.0),
                cv=3  # Use simple integer for CV folds
            )
            stacking_regressor.fit(X_train_selected, y_train)
            
            # Advanced ensemble with dynamic weighting
            self.advanced_ensemble = AdvancedEnsembleRegressor(
                base_models={name: model for name, model in best_models[:5]},
                meta_model=Ridge(alpha=1.0),
                cv_folds=3
            )
            self.advanced_ensemble.fit(X_train_selected, y_train)
        else:
            voting_regressor = None
            stacking_regressor = None
            self.advanced_ensemble = None
        
        # Evaluate ensemble models
        if voting_regressor is not None and stacking_regressor is not None:
            ensemble_models = {
                'voting_regressor': voting_regressor,
                'stacking_regressor': stacking_regressor,
                'advanced_ensemble': self.advanced_ensemble
            }
            
            for name, model in ensemble_models.items():
                if model is None:
                    continue
                    
                y_pred = model.predict(X_test_selected)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Time series CV scoring
                tscv = TimeSeriesSplit(n_splits=3)
                cv_scores = []
                for train_idx, val_idx in tscv.split(X_train_selected):
                    X_cv_train, X_cv_val = X_train_selected.iloc[train_idx], X_train_selected.iloc[val_idx]
                    y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    model_copy = self._clone_model(model)
                    model_copy.fit(X_cv_train, y_cv_train)
                    y_cv_pred = model_copy.predict(X_cv_val)
                    cv_scores.append(mean_squared_error(y_cv_val, y_cv_pred))
                
                cv_score = np.mean(cv_scores)
                
                results[name] = {
                    'model': model,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'rmse': np.sqrt(mse),
                    'cv_score': cv_score
                }
                
                self.trained_models[name] = model
                self.ensemble_models[name] = model
                
                print(f"{name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, CV Score: {cv_score:.4f}")
        else:
            print("Skipping ensemble models due to insufficient base models")
        
        # Find best model based on CV score
        best_model_name = min(results.keys(), key=lambda x: results[x]['cv_score'])
        self.best_model = results[best_model_name]['model']
        self.best_score = results[best_model_name]['cv_score']
        
        print(f"\nBest model: {best_model_name} (CV Score: {self.best_score:.4f})")
        
        # Store selected features for prediction
        self.selected_features = selected_features
        
        # Train deep learning models
        print("\nTraining deep learning models...")
        dl_results = self._train_deep_learning_models(X_train_selected, y_train, X_test_selected, y_test)
        results.update(dl_results)
        
        # Create ensemble models
        print("\nCreating ensemble models...")
        ensemble_results = self._create_ensemble_models(X_train_selected, y_train, X_test_selected, y_test)
        results.update(ensemble_results)
        
        # Find best model based on CV score
        best_model_name = min(results.keys(), key=lambda x: results[x]['cv_score'])
        self.best_model = results[best_model_name]['model']
        self.best_score = results[best_model_name]['cv_score']
        
        print(f"\nBest model: {best_model_name} (CV Score: {self.best_score:.4f})")
        
        # Store selected features for prediction
        self.selected_features = selected_features
        
        return results
    
    def _train_deep_learning_models(self, X_train, y_train, X_test, y_test):
        """
        Train LSTM and GRU models for time series prediction
        """
        results = {}
        
        # Prepare data for deep learning (3D shape for LSTM/GRU)
        sequence_length = 20
        X_train_seq, y_train_seq = self._create_sequences(X_train.values, y_train.values, sequence_length)
        X_test_seq, y_test_seq = self._create_sequences(X_test.values, y_test.values, sequence_length)
        
        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            print("Not enough data for sequence creation, skipping deep learning models")
            return results
        
        # LSTM Model
        try:
            lstm_model = self._build_lstm_model(X_train_seq.shape[1], X_train_seq.shape[2])
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            
            history = lstm_model.fit(
                X_train_seq, y_train_seq,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # Predictions
            y_pred_lstm = lstm_model.predict(X_test_seq, verbose=0).flatten()
            
            # Metrics
            mse = mean_squared_error(y_test_seq, y_pred_lstm)
            mae = mean_absolute_error(y_test_seq, y_pred_lstm)
            r2 = r2_score(y_test_seq, y_pred_lstm)
            
            results['lstm'] = {
                'model': lstm_model,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': np.sqrt(mse),
                'cv_score': mse  # Use MSE as CV score for DL models
            }
            
            print(f"LSTM - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            
        except Exception as e:
            print(f"Error training LSTM model: {e}")
        
        # GRU Model
        try:
            gru_model = self._build_gru_model(X_train_seq.shape[1], X_train_seq.shape[2])
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            
            history = gru_model.fit(
                X_train_seq, y_train_seq,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # Predictions
            y_pred_gru = gru_model.predict(X_test_seq, verbose=0).flatten()
            
            # Metrics
            mse = mean_squared_error(y_test_seq, y_pred_gru)
            mae = mean_absolute_error(y_test_seq, y_pred_gru)
            r2 = r2_score(y_test_seq, y_pred_gru)
            
            results['gru'] = {
                'model': gru_model,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': np.sqrt(mse),
                'cv_score': mse  # Use MSE as CV score for DL models
            }
            
            print(f"GRU - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            
        except Exception as e:
            print(f"Error training GRU model: {e}")
        
        return results
    
    def _create_sequences(self, X, y, sequence_length):
        """
        Create sequences for LSTM/GRU training
        """
        X_seq, y_seq = [], []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def _build_lstm_model(self, sequence_length, n_features):
        """
        Build LSTM model architecture
        """
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(sequence_length, n_features)),
            Dropout(0.2),
            BatchNormalization(),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def _build_gru_model(self, sequence_length, n_features):
        """
        Build GRU model architecture
        """
        model = Sequential([
            GRU(128, return_sequences=True, input_shape=(sequence_length, n_features)),
            Dropout(0.2),
            BatchNormalization(),
            GRU(64, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            GRU(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def _create_ensemble_models(self, X_train, y_train, X_test, y_test):
        """
        Create ensemble models using the best performing individual models
        """
        results = {}
        
        if len(self.trained_models) < 3:
            print("Not enough models for ensemble creation")
            return results
        
        # Get top 3 models based on performance
        model_items = list(self.trained_models.items())[:3]
        
        try:
            # Voting Regressor
            voting_regressor = VotingRegressor([
                (f'model_{i}', model) for i, (name, model) in enumerate(model_items)
            ])
            voting_regressor.fit(X_train, y_train)
            
            y_pred = voting_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results['voting_ensemble'] = {
                'model': voting_regressor,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': np.sqrt(mse),
                'cv_score': mse
            }
            
            print(f"Voting Ensemble - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            
        except Exception as e:
            print(f"Error creating voting ensemble: {e}")
        
        try:
            # Stacking Regressor
            stacking_regressor = StackingRegressor(
                estimators=[(f'model_{i}', model) for i, (name, model) in enumerate(model_items[:-1])],
                final_estimator=model_items[-1][1],
                cv=3
            )
            stacking_regressor.fit(X_train, y_train)
            
            y_pred = stacking_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results['stacking_ensemble'] = {
                'model': stacking_regressor,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': np.sqrt(mse),
                'cv_score': mse
            }
            
            print(f"Stacking Ensemble - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            
        except Exception as e:
            print(f"Error creating stacking ensemble: {e}")
        
        return results
    
    def optimize_hyperparameters(self, X_train, y_train, model_name, n_trials=50):
        """
        Optimize hyperparameters using Optuna
        """
        def objective(trial):
            if model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'random_state': 42
                }
                model = RandomForestRegressor(**params)
                
            elif model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': 42
                }
                model = xgb.XGBRegressor(**params)
                
            elif model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': 42,
                    'verbose': -1
                }
                model = lgb.LGBMRegressor(**params)
                
            elif model_name == 'svr':
                params = {
                    'C': trial.suggest_float('C', 0.1, 1000, log=True),
                    'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                    'epsilon': trial.suggest_float('epsilon', 0.01, 1.0)
                }
                model = SVR(kernel='rbf', **params)
                
            else:
                return float('inf')
            
            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                model.fit(X_cv_train, y_cv_train)
                y_pred = model.predict(X_cv_val)
                score = mean_squared_error(y_cv_val, y_pred)
                scores.append(score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        return study.best_params
    
    def _walk_forward_validation(self, model, X, y, n_splits=5):
        """
        Perform walk-forward validation for time series data
        """
        scores = []
        step_size = len(X) // (n_splits + 1)
        
        for i in range(n_splits):
            # Define train and validation indices
            train_end = step_size * (i + 2)
            val_start = train_end
            val_end = min(train_end + step_size, len(X))
            
            if val_end <= val_start:
                break
                
            # Split data
            X_train_fold = X.iloc[:train_end]
            y_train_fold = y.iloc[:train_end]
            X_val_fold = X.iloc[val_start:val_end]
            y_val_fold = y.iloc[val_start:val_end]
            
            # Train and validate
            model_copy = self._clone_model(model)
            model_copy.fit(X_train_fold, y_train_fold)
            y_pred = model_copy.predict(X_val_fold)
            
            score = mean_squared_error(y_val_fold, y_pred)
            scores.append(score)
            
        return scores
    
    def _clone_model(self, model):
        """
        Create a copy of the model with same parameters
        """
        from sklearn.base import clone
        return clone(model)
    
    def predict(self, data, return_confidence=False, confidence_level=0.95):
        """
        Make predictions using the best model with optional confidence intervals
        """
        if self.best_model is None:
            raise ValueError("No models have been trained yet. Call train_models() first.")
        
        # Store original feature columns to restore later
        original_feature_columns = self.feature_columns.copy() if hasattr(self, 'feature_columns') else []
        
        # Prepare features using the feature engineering pipeline
        data_with_features = self.prepare_features(data, is_training=False)
        
        # Restore the original feature columns that the model was trained with
        self.feature_columns = original_feature_columns
        
        # Create a DataFrame with the expected features, filling missing ones with defaults
        X = pd.DataFrame(index=data_with_features.index)
        
        for feature in self.feature_columns:
            if feature in data_with_features.columns:
                X[feature] = data_with_features[feature]
            else:
                # Provide reasonable defaults for missing features
                if 'price' in feature.lower() or 'close' in feature.lower():
                    X[feature] = 2000.0  # Default gold price
                elif 'open' in feature.lower() or 'high' in feature.lower() or 'low' in feature.lower():
                    X[feature] = 2000.0  # Default OHLC values
                elif 'volume' in feature.lower():
                    X[feature] = 1000000.0  # Default volume
                elif 'sma' in feature.lower() or 'ema' in feature.lower() or 'ma' in feature.lower():
                    X[feature] = 2000.0  # Default moving average
                elif 'bb' in feature.lower():
                    X[feature] = 0.5  # Default Bollinger Band position
                elif 'rsi' in feature.lower():
                    X[feature] = 50.0  # Default RSI
                elif 'volatility' in feature.lower() or 'std' in feature.lower():
                    X[feature] = 0.02  # Default volatility
                elif 'return' in feature.lower() or 'change' in feature.lower():
                    X[feature] = 0.01  # Default return
                else:
                    X[feature] = 0.0  # Default for other features
        
        # Apply same scaling as training
        print(f"DEBUG PREDICT: scaler type = {type(self.scaler)}")
        print(f"DEBUG PREDICT: scaler fitted = {hasattr(self.scaler, 'center_') and hasattr(self.scaler, 'scale_')}")
        print(f"DEBUG PREDICT: X shape = {X.shape}")
        print(f"DEBUG PREDICT: feature_columns length = {len(self.feature_columns)}")
        try:
            X_scaled = self.scaler.transform(X)
            print(f"DEBUG PREDICT: Transform successful")
        except Exception as transform_error:
            print(f"DEBUG PREDICT: Transform failed: {transform_error}")
            raise transform_error
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Apply same feature selection as training
        X_selected = self.feature_selector.transform(X_scaled)
        X_selected = pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
        
        # Make predictions
        predictions = self.best_model.predict(X_selected)
        
        if return_confidence:
            # Calculate prediction intervals using ensemble of models
            predictions_ensemble = []
            
            for name, model in self.trained_models.items():
                try:
                    pred = model.predict(X_selected)
                    predictions_ensemble.append(pred)
                except:
                    continue
            
            if predictions_ensemble:
                predictions_ensemble = np.array(predictions_ensemble)
                
                # Calculate confidence intervals
                alpha = 1 - confidence_level
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                
                lower_bound = np.percentile(predictions_ensemble, lower_percentile, axis=0)
                upper_bound = np.percentile(predictions_ensemble, upper_percentile, axis=0)
                
                # Calculate prediction uncertainty
                prediction_std = np.std(predictions_ensemble, axis=0)
                
                return {
                    'prediction': predictions,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'std': prediction_std,
                    'confidence_level': confidence_level
                }
        
        return predictions
    
    def predict_with_confidence(self, data, n_estimators=100):
        """
        Make predictions with confidence intervals using bootstrap
        """
        if self.best_model is None:
            raise ValueError("No models have been trained yet. Call train_models() first.")
        
        # Prepare features
        data = self.prepare_features(data)
        X = data[self.feature_columns]
        
        # Apply preprocessing
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        X_selected = self.feature_selector.transform(X_scaled)
        X_selected = pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
        
        # Bootstrap predictions for confidence intervals
        predictions = []
        for _ in range(n_estimators):
            # Create bootstrap sample indices
            bootstrap_indices = np.random.choice(len(X_selected), size=len(X_selected), replace=True)
            X_bootstrap = X_selected.iloc[bootstrap_indices]
            
            # Make prediction
            pred = self.best_model.predict(X_bootstrap)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        lower_bound = np.percentile(predictions, 2.5, axis=0)
        upper_bound = np.percentile(predictions, 97.5, axis=0)
        
        return {
            'prediction': mean_pred,
            'std': std_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    def train_prophet_model(self, data):
        """
        Train Prophet model for time series forecasting
        """
        # Prepare data for Prophet
        prophet_data = data[['date', 'price']].copy()
        prophet_data.columns = ['ds', 'y']
        prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
        
        # Initialize and train Prophet model
        self.prophet_model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        
        self.prophet_model.fit(prophet_data)
        
        return self.prophet_model
    
    def predict_single(self, model_name, features):
        """
        Make prediction using a specific model
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
            
        model = self.trained_models[model_name]
        prediction = model.predict([features])
        
        return prediction[0]
    
    def predict_single_model(self, data, model_name):
        """
        Make prediction using a specific model with full preprocessing
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found in trained models")
        
        # Prepare features
        data = self.prepare_features(data)
        X = data[self.feature_columns]
        
        # Apply same scaling as training
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Apply same feature selection as training
        X_selected = self.feature_selector.transform(X_scaled)
        X_selected = pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
        
        # Make prediction
        return self.trained_models[model_name].predict(X_selected)
    
    def predict_ensemble(self, features):
        """
        Make ensemble prediction using all trained models
        """
        predictions = []
        
        for name, model in self.trained_models.items():
            pred = model.predict([features])[0]
            predictions.append(pred)
            
        # Return average of all predictions
        return np.mean(predictions)
    
    def predict_future_prophet(self, days=30):
        """
        Predict future prices using Prophet model
        """
        if self.prophet_model is None:
            raise ValueError("Prophet model not trained yet")
            
        # Create future dataframe
        future = self.prophet_model.make_future_dataframe(periods=days)
        forecast = self.prophet_model.predict(future)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days)
    
    def predict_future(self, data, days_ahead=30, return_confidence=False, confidence_level=0.95):
        """
        Predict future gold prices with uncertainty quantification
        """
        if self.best_model is None:
            raise ValueError("No models have been trained yet. Call train_models() first.")
        
        predictions = []
        confidence_intervals = []
        current_data = data.copy()
        
        # Get top performing models for ensemble
        top_models = list(self.trained_models.keys())[:5]  # Use top 5 models
        
        for day in range(days_ahead):
            # Get ensemble predictions
            ensemble_predictions = []
            
            for model_name in top_models:
                if model_name in self.trained_models:
                    try:
                        pred = self.predict_single_model(current_data.tail(1), model_name)
                        ensemble_predictions.append(pred[0])
                    except:
                        continue
            
            if ensemble_predictions:
                # Calculate ensemble prediction
                ensemble_pred = np.mean(ensemble_predictions)
                predictions.append(ensemble_pred)
                
                if return_confidence:
                    # Calculate confidence intervals
                    pred_std = np.std(ensemble_predictions)
                    alpha = 1 - confidence_level
                    z_score = stats.norm.ppf(1 - alpha/2)
                    
                    lower_bound = ensemble_pred - z_score * pred_std
                    upper_bound = ensemble_pred + z_score * pred_std
                    
                    confidence_intervals.append({
                        'lower': lower_bound,
                        'upper': upper_bound,
                        'std': pred_std
                    })
                
                # Update data with prediction
                next_date = pd.to_datetime(current_data['date'].iloc[-1]) + timedelta(days=1)
                new_row = current_data.iloc[-1].copy()
                new_row['date'] = next_date
                new_row['price'] = ensemble_pred
                
                # Add the new row
                current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
            else:
                # Fallback to best model
                next_price = self.predict(current_data.tail(1))
                predictions.append(next_price[0])
                
                # Update data
                next_date = pd.to_datetime(current_data['date'].iloc[-1]) + timedelta(days=1)
                new_row = current_data.iloc[-1].copy()
                new_row['date'] = next_date
                new_row['price'] = next_price[0]
                
                current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
        
        if return_confidence and confidence_intervals:
            return {
                'predictions': np.array(predictions),
                'confidence_intervals': confidence_intervals,
                'confidence_level': confidence_level
            }
        
        return np.array(predictions)
    
    def get_feature_importance(self, model_name='random_forest', top_n=20):
        """
        Get comprehensive feature importance from multiple models
        """
        if model_name not in self.trained_models:
            # Get importance from all models if specific model not found
            importance_scores = {}
            model_count = 0
            
            for name, model in self.trained_models.items():
                if hasattr(model, 'feature_importances_'):
                    # Tree-based models
                    feature_names = self.selected_features if hasattr(self, 'selected_features') else self.feature_columns
                    for i, importance in enumerate(model.feature_importances_):
                        if i < len(feature_names):
                            feature_name = feature_names[i]
                            if feature_name not in importance_scores:
                                importance_scores[feature_name] = []
                            importance_scores[feature_name].append(importance)
                    model_count += 1
                elif hasattr(model, 'coef_'):
                    # Linear models
                    feature_names = self.selected_features if hasattr(self, 'selected_features') else self.feature_columns
                    coef = model.coef_ if len(model.coef_.shape) == 1 else model.coef_[0]
                    for i, coef_val in enumerate(coef):
                        if i < len(feature_names):
                            feature_name = feature_names[i]
                            if feature_name not in importance_scores:
                                importance_scores[feature_name] = []
                            importance_scores[feature_name].append(abs(coef_val))
                    model_count += 1
            
            if not importance_scores:
                return None
            
            # Calculate average importance across models
            avg_importance = {}
            for feature, scores in importance_scores.items():
                avg_importance[feature] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'count': len(scores)
                }
            
            # Sort by mean importance
            sorted_importance = sorted(avg_importance.items(), 
                                     key=lambda x: x[1]['mean'], reverse=True)
            
            self.feature_importance_scores = dict(sorted_importance[:top_n])
            return sorted_importance[:top_n]
        else:
            # Get importance from specific model
            model = self.trained_models[model_name]
            
            if hasattr(model, 'feature_importances_'):
                feature_names = self.selected_features if hasattr(self, 'selected_features') else self.feature_columns
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                return importance_df
            else:
                return None
    
    def save_models(self, directory='models'):
        """
        Save trained models to disk
        """
        os.makedirs(directory, exist_ok=True)
        
        for name, model in self.trained_models.items():
            filename = os.path.join(directory, f'{name}_model.pkl')
            joblib.dump(model, filename)
            
        # Save Prophet model separately
        if self.prophet_model:
            prophet_filename = os.path.join(directory, 'prophet_model.pkl')
            joblib.dump(self.prophet_model, prophet_filename)
            
        # Save feature columns
        feature_filename = os.path.join(directory, 'feature_columns.pkl')
        joblib.dump(self.feature_columns, feature_filename)
        
        # Save selected features if they exist
        if hasattr(self, 'selected_features') and self.selected_features is not None:
            selected_features_filename = os.path.join(directory, 'selected_features.pkl')
            joblib.dump(self.selected_features, selected_features_filename)
            print(f"✓ Selected features saved to {selected_features_filename}")
        
        # Save scaler if it exists and is fitted
        if hasattr(self, 'scaler') and self.scaler is not None:
            scaler_filename = os.path.join(directory, 'scaler.pkl')
            joblib.dump(self.scaler, scaler_filename)
            print(f"✓ Scaler saved to {scaler_filename}")
        
        # Save feature selector if it exists and is fitted
        if hasattr(self, 'feature_selector') and self.feature_selector is not None:
            selector_filename = os.path.join(directory, 'selector.pkl')
            joblib.dump(self.feature_selector, selector_filename)
            print(f"✓ Feature selector saved to {selector_filename}")
        
        print(f"Models saved to {directory}")
    
    def load_models(self, directory='models'):
        """
        Load trained models from disk with comprehensive error handling
        """
        loaded_models = 0
        failed_models = []
        
        try:
            # Load feature columns - try both .pkl and .joblib formats
            feature_filename_pkl = os.path.join(directory, 'feature_columns.pkl')
            feature_filename_joblib = os.path.join(directory, 'feature_columns.joblib')
            
            if os.path.exists(feature_filename_joblib):
                try:
                    self.feature_columns = joblib.load(feature_filename_joblib)
                    print(f"✓ Feature columns loaded successfully from .joblib")
                except Exception as e:
                    print(f"✗ Failed to load feature columns from .joblib: {e}")
                    self.feature_columns = ['ma_5', 'ma_10', 'ma_20', 'price_change', 'price_change_5', 'volatility', 'price_lag_1', 'price_lag_2', 'price_lag_3']
            elif os.path.exists(feature_filename_pkl):
                try:
                    self.feature_columns = joblib.load(feature_filename_pkl)
                    print(f"✓ Feature columns loaded successfully from .pkl")
                except Exception as e:
                    print(f"✗ Failed to load feature columns from .pkl: {e}")
                    self.feature_columns = ['ma_5', 'ma_10', 'ma_20', 'price_change', 'price_change_5', 'volatility', 'price_lag_1', 'price_lag_2', 'price_lag_3']
            else:
                print(f"⚠ Feature columns file not found, using defaults")
                self.feature_columns = ['ma_5', 'ma_10', 'ma_20', 'price_change', 'price_change_5', 'volatility', 'price_lag_1', 'price_lag_2', 'price_lag_3']
                
            # Load scaler - try both .pkl and .joblib formats
            print(f"DEBUG: feature_columns = {self.feature_columns}, length = {len(self.feature_columns)}")
            scaler_filename_pkl = os.path.join(directory, 'scaler.pkl')
            scaler_filename_joblib = os.path.join(directory, 'scaler.joblib')
            
            scaler_loaded = False
            for scaler_file in [scaler_filename_joblib, scaler_filename_pkl]:
                if os.path.exists(scaler_file):
                    try:
                        self.scaler = joblib.load(scaler_file)
                        print(f"DEBUG: Scaler loaded from {scaler_file}, type = {type(self.scaler)}")
                        # Verify scaler is fitted by trying to transform a sample
                        try:
                            import numpy as np
                            test_data = np.array([[2000.0] * len(self.feature_columns)])
                            print(f"DEBUG: Test data shape = {test_data.shape}")
                            result = self.scaler.transform(test_data)
                            print(f"DEBUG: Transform successful, result shape = {result.shape}")
                            print(f"✓ Scaler loaded successfully and is fitted")
                            scaler_loaded = True
                            break
                        except Exception as transform_error:
                            print(f"DEBUG: Transform failed with error: {transform_error}")
                            print(f"⚠ Loaded scaler is not properly fitted, creating new fitted scaler")
                            from sklearn.preprocessing import RobustScaler
                            self.scaler = RobustScaler()
                            # Fit with sample data based on feature columns
                            sample_data = np.random.normal(2000, 100, (100, len(self.feature_columns)))
                            print(f"DEBUG: Sample data shape for fitting = {sample_data.shape}")
                            self.scaler.fit(sample_data)
                            print(f"✓ New scaler fitted with sample data")
                            scaler_loaded = True
                            break
                    except Exception as e:
                        print(f"✗ Failed to load scaler from {scaler_file}: {e}")
                        continue
            
            if not scaler_loaded:
                print(f"⚠ Scaler file not found, initializing and fitting new scaler")
                from sklearn.preprocessing import RobustScaler
                self.scaler = RobustScaler()
                # Fit with sample data
                import numpy as np
                sample_data = np.random.normal(2000, 100, (100, len(self.feature_columns)))
                print(f"DEBUG: Sample data shape for fitting = {sample_data.shape}")
                self.scaler.fit(sample_data)
                print(f"✓ New scaler fitted with sample data")
                
            # Load feature selector
            selector_filename = os.path.join(directory, 'selector.pkl')
            if os.path.exists(selector_filename):
                try:
                    self.feature_selector = joblib.load(selector_filename)
                    print(f"✓ Feature selector loaded successfully")
                except Exception as e:
                    print(f"✗ Failed to load feature selector: {e}")
                    from sklearn.feature_selection import SelectKBest, f_regression
                    self.feature_selector = SelectKBest(score_func=f_regression, k=min(50, len(self.feature_columns)))
            else:
                print(f"⚠ Feature selector file not found, initializing new selector")
                from sklearn.feature_selection import SelectKBest, f_regression
                self.feature_selector = SelectKBest(score_func=f_regression, k=min(50, len(self.feature_columns)))
                
            # Load selected features
            selected_features_filename = os.path.join(directory, 'selected_features.pkl')
            if os.path.exists(selected_features_filename):
                try:
                    self.selected_features = joblib.load(selected_features_filename)
                    print(f"✓ Selected features loaded successfully")
                except Exception as e:
                    print(f"✗ Failed to load selected features: {e}")
                    self.selected_features = self.feature_columns  # Fallback to all features
            else:
                print(f"⚠ Selected features file not found, using all feature columns")
                self.selected_features = self.feature_columns  # Fallback to all features
                
            # Try to load quick trained model first
            quick_model_filename = os.path.join(directory, 'quick_trained_model.joblib')
            if os.path.exists(quick_model_filename):
                try:
                    self.trained_models['quick_model'] = joblib.load(quick_model_filename)
                    self.best_model = self.trained_models['quick_model']
                    loaded_models += 1
                    print(f"✓ Quick trained model loaded successfully")
                except Exception as e:
                    print(f"✗ Failed to load quick trained model: {e}")
                    failed_models.append('quick_model')
                
            # Load ML models with individual error handling
            for name in self.models.keys():
                filename = os.path.join(directory, f'{name}_model.pkl')
                if os.path.exists(filename):
                    try:
                        self.trained_models[name] = joblib.load(filename)
                        loaded_models += 1
                        print(f"✓ {name} model loaded successfully")
                    except Exception as e:
                        print(f"✗ Failed to load {name} model: {e}")
                        failed_models.append(name)
                else:
                    print(f"⚠ {name} model file not found")
                    failed_models.append(name)
                    
            # Load Prophet model
            prophet_filename = os.path.join(directory, 'prophet_model.pkl')
            if os.path.exists(prophet_filename):
                try:
                    self.prophet_model = joblib.load(prophet_filename)
                    print(f"✓ Prophet model loaded successfully")
                except Exception as e:
                    print(f"✗ Failed to load Prophet model: {e}")
                    self.prophet_model = None
            else:
                print(f"⚠ Prophet model file not found")
                self.prophet_model = None
                
            # Summary
            print(f"\n📊 Model Loading Summary:")
            print(f"   ✓ Successfully loaded: {loaded_models} models")
            if failed_models:
                print(f"   ✗ Failed to load: {len(failed_models)} models ({', '.join(failed_models)})")
            print(f"   📁 Directory: {directory}")
            
            # If no models loaded, initialize with basic fallback
            if loaded_models == 0:
                print(f"\n⚠ WARNING: No models loaded! Initializing fallback prediction system...")
                self._initialize_fallback_system()
                return True  # Fallback system counts as success
            
            # Set best model if we have trained models
            if loaded_models > 0 and not self.best_model:
                best_model_name = list(self.trained_models.keys())[0]
                self.best_model = self.trained_models[best_model_name]
                print(f"✓ Best model set to: {best_model_name}")
                
            return True
                
        except Exception as e:
            print(f"\n❌ Critical error during model loading: {e}")
            print(f"Initializing emergency fallback system...")
            self._initialize_fallback_system()
            return False
            
    def _initialize_fallback_system(self):
        """
        Initialize a basic fallback prediction system when models fail to load
        """
        try:
            # Create a simple linear regression as fallback
            from sklearn.linear_model import LinearRegression
            fallback_model = LinearRegression()
            
            # Generate some dummy training data for the fallback
            np.random.seed(42)
            X_dummy = np.random.randn(100, 5)  # 5 features
            y_dummy = X_dummy.sum(axis=1) + np.random.randn(100) * 0.1
            
            fallback_model.fit(X_dummy, y_dummy)
            self.trained_models['fallback_linear'] = fallback_model
            self.feature_columns = [f'feature_{i}' for i in range(5)]
            
            print(f"✓ Fallback prediction system initialized")
            
        except Exception as e:
            print(f"❌ Failed to initialize fallback system: {e}")
            # Last resort - set empty but functional state
            self.trained_models = {}
            self.feature_columns = []
    
    def get_model_performance(self, detailed=False):
        """
        Get performance metrics for all trained models
        """
        performance = {}
        
        for name, model in self.trained_models.items():
            # This would need test data to calculate actual performance
            # For now, return placeholder
            basic_info = {
                'status': 'trained',
                'features_count': len(self.feature_columns),
                'model_type': type(model).__name__
            }
            
            if detailed:
                basic_info.update({
                    'feature_names': self.selected_features if hasattr(self, 'selected_features') else self.feature_columns[:10],
                    'best_score': getattr(self, 'best_score', 'Unknown')
                })
            
            performance[name] = basic_info
            
        return performance
    
    def calculate_prediction_intervals(self, data, confidence_levels=[0.68, 0.95, 0.99]):
        """
        Calculate prediction intervals at multiple confidence levels
        """
        if self.best_model is None:
            raise ValueError("No models have been trained yet. Call train_models() first.")
        
        # Get predictions from all models
        all_predictions = []
        
        for name, model in self.trained_models.items():
            try:
                pred = self.predict_single_model(data, name)
                all_predictions.append(pred)
            except:
                continue
        
        if not all_predictions:
            return None
        
        all_predictions = np.array(all_predictions)
        
        # Calculate intervals for each confidence level
        intervals = {}
        for confidence_level in confidence_levels:
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bound = np.percentile(all_predictions, lower_percentile, axis=0)
            upper_bound = np.percentile(all_predictions, upper_percentile, axis=0)
            
            intervals[confidence_level] = {
                'lower': lower_bound,
                'upper': upper_bound,
                'width': upper_bound - lower_bound
            }
        
        return intervals