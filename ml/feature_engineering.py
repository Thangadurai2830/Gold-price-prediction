import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Advanced feature engineering for gold price prediction
    """
    
    def __init__(self):
        self.scaler = None
        self.feature_selector = None
        self.pca = None
        self.selected_features = []
        
    def create_lag_features(self, data, target_column='price', lags=[1, 2, 3, 5, 10, 20]):
        """
        Create lag features for time series data
        """
        data_copy = data.copy()
        
        for lag in lags:
            data_copy[f'{target_column}_lag_{lag}'] = data_copy[target_column].shift(lag)
            
        print(f"Created {len(lags)} lag features")
        return data_copy
    
    def create_rolling_features(self, data, target_column='price', windows=[5, 10, 20, 50]):
        """
        Create rolling statistical features
        """
        data_copy = data.copy()
        
        for window in windows:
            # Rolling mean
            data_copy[f'{target_column}_rolling_mean_{window}'] = data_copy[target_column].rolling(window=window).mean()
            
            # Rolling standard deviation
            data_copy[f'{target_column}_rolling_std_{window}'] = data_copy[target_column].rolling(window=window).std()
            
            # Rolling min and max
            data_copy[f'{target_column}_rolling_min_{window}'] = data_copy[target_column].rolling(window=window).min()
            data_copy[f'{target_column}_rolling_max_{window}'] = data_copy[target_column].rolling(window=window).max()
            
            # Rolling median
            data_copy[f'{target_column}_rolling_median_{window}'] = data_copy[target_column].rolling(window=window).median()
            
            # Rolling quantiles
            data_copy[f'{target_column}_rolling_q25_{window}'] = data_copy[target_column].rolling(window=window).quantile(0.25)
            data_copy[f'{target_column}_rolling_q75_{window}'] = data_copy[target_column].rolling(window=window).quantile(0.75)
            
        print(f"Created rolling features for {len(windows)} windows")
        return data_copy
    
    def create_momentum_features(self, data, target_column='price'):
        """
        Create momentum and rate of change features
        """
        data_copy = data.copy()
        
        # Rate of change for different periods
        for period in [1, 5, 10, 20]:
            data_copy[f'{target_column}_roc_{period}'] = data_copy[target_column].pct_change(period)
            
        # Momentum (current price - price n periods ago)
        for period in [5, 10, 20]:
            data_copy[f'{target_column}_momentum_{period}'] = data_copy[target_column] - data_copy[target_column].shift(period)
            
        # Acceleration (rate of change of momentum)
        data_copy[f'{target_column}_acceleration'] = data_copy[f'{target_column}_roc_1'].diff()
        
        print("Created momentum features")
        return data_copy
    
    def create_volatility_features(self, data, target_column='price'):
        """
        Create volatility-based features
        """
        data_copy = data.copy()
        
        # Historical volatility for different windows
        for window in [5, 10, 20, 50]:
            returns = data_copy[target_column].pct_change()
            data_copy[f'{target_column}_volatility_{window}'] = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
            
        # Volatility ratio (short-term vs long-term)
        data_copy[f'{target_column}_vol_ratio_5_20'] = data_copy[f'{target_column}_volatility_5'] / data_copy[f'{target_column}_volatility_20']
        data_copy[f'{target_column}_vol_ratio_10_50'] = data_copy[f'{target_column}_volatility_10'] / data_copy[f'{target_column}_volatility_50']
        
        # Volatility of volatility
        data_copy[f'{target_column}_vol_of_vol'] = data_copy[f'{target_column}_volatility_20'].rolling(window=20).std()
        
        print("Created volatility features")
        return data_copy
    
    def create_price_position_features(self, data, target_column='price'):
        """
        Create features based on price position relative to historical levels
        """
        data_copy = data.copy()
        
        for window in [20, 50, 100]:
            # Price position within rolling window (0 = at minimum, 1 = at maximum)
            rolling_min = data_copy[target_column].rolling(window=window).min()
            rolling_max = data_copy[target_column].rolling(window=window).max()
            data_copy[f'{target_column}_position_{window}'] = (
                (data_copy[target_column] - rolling_min) / (rolling_max - rolling_min)
            )
            
            # Distance from rolling mean (normalized)
            rolling_mean = data_copy[target_column].rolling(window=window).mean()
            rolling_std = data_copy[target_column].rolling(window=window).std()
            data_copy[f'{target_column}_zscore_{window}'] = (
                (data_copy[target_column] - rolling_mean) / rolling_std
            )
            
        print("Created price position features")
        return data_copy
    
    def create_trend_features(self, data, target_column='price'):
        """
        Create trend-based features
        """
        data_copy = data.copy()
        
        # Simple trend indicators
        for window in [5, 10, 20]:
            # Trend direction (1 if uptrend, -1 if downtrend, 0 if sideways)
            sma = data_copy[target_column].rolling(window=window).mean()
            data_copy[f'{target_column}_trend_{window}'] = np.where(
                data_copy[target_column] > sma, 1,
                np.where(data_copy[target_column] < sma, -1, 0)
            )
            
            # Trend strength (slope of linear regression)
            def calculate_slope(series):
                if len(series) < 2:
                    return 0
                x = np.arange(len(series))
                slope = np.polyfit(x, series, 1)[0]
                return slope
            
            data_copy[f'{target_column}_slope_{window}'] = (
                data_copy[target_column].rolling(window=window).apply(calculate_slope, raw=False)
            )
            
        print("Created trend features")
        return data_copy
    
    def create_cyclical_features(self, data, date_column='date'):
        """
        Create cyclical features from date information
        """
        data_copy = data.copy()
        
        if date_column in data_copy.columns:
            data_copy[date_column] = pd.to_datetime(data_copy[date_column])
            
            # Extract date components
            data_copy['year'] = data_copy[date_column].dt.year
            data_copy['month'] = data_copy[date_column].dt.month
            data_copy['day'] = data_copy[date_column].dt.day
            data_copy['day_of_week'] = data_copy[date_column].dt.dayofweek
            data_copy['day_of_year'] = data_copy[date_column].dt.dayofyear
            data_copy['week_of_year'] = data_copy[date_column].dt.isocalendar().week
            data_copy['quarter'] = data_copy[date_column].dt.quarter
            
            # Create cyclical features (sine and cosine transformations)
            data_copy['month_sin'] = np.sin(2 * np.pi * data_copy['month'] / 12)
            data_copy['month_cos'] = np.cos(2 * np.pi * data_copy['month'] / 12)
            
            data_copy['day_of_week_sin'] = np.sin(2 * np.pi * data_copy['day_of_week'] / 7)
            data_copy['day_of_week_cos'] = np.cos(2 * np.pi * data_copy['day_of_week'] / 7)
            
            data_copy['day_of_year_sin'] = np.sin(2 * np.pi * data_copy['day_of_year'] / 365.25)
            data_copy['day_of_year_cos'] = np.cos(2 * np.pi * data_copy['day_of_year'] / 365.25)
            
            # Market-specific features
            data_copy['is_month_end'] = (data_copy[date_column].dt.day >= 28).astype(int)
            data_copy['is_quarter_end'] = data_copy[date_column].dt.month.isin([3, 6, 9, 12]).astype(int)
            data_copy['is_year_end'] = (data_copy[date_column].dt.month == 12).astype(int)
            
            print("Created cyclical features")
        else:
            print(f"Date column '{date_column}' not found")
            
        return data_copy
    
    def create_interaction_features(self, data, feature_pairs=None):
        """
        Create interaction features between important variables
        """
        data_copy = data.copy()
        
        if feature_pairs is None:
            # Default important feature pairs for gold prediction
            numeric_columns = data_copy.select_dtypes(include=[np.number]).columns
            important_features = [col for col in numeric_columns if any(keyword in col.lower() 
                                for keyword in ['price', 'volume', 'usd', 'sp500', 'treasury', 'oil'])]
            
            if len(important_features) >= 2:
                feature_pairs = [(important_features[i], important_features[j]) 
                               for i in range(len(important_features)) 
                               for j in range(i+1, min(i+3, len(important_features)))]  # Limit pairs
        
        if feature_pairs:
            for feat1, feat2 in feature_pairs[:10]:  # Limit to 10 interactions
                if feat1 in data_copy.columns and feat2 in data_copy.columns:
                    # Multiplication interaction
                    data_copy[f'{feat1}_x_{feat2}'] = data_copy[feat1] * data_copy[feat2]
                    
                    # Ratio interaction (avoid division by zero)
                    data_copy[f'{feat1}_div_{feat2}'] = data_copy[feat1] / (data_copy[feat2] + 1e-8)
                    
            print(f"Created {len(feature_pairs)} interaction features")
        
        return data_copy
    
    def scale_features(self, data, method='standard', exclude_columns=['date', 'price']):
        """
        Scale numerical features
        """
        data_copy = data.copy()
        
        # Select numerical columns to scale
        numeric_columns = data_copy.select_dtypes(include=[np.number]).columns
        columns_to_scale = [col for col in numeric_columns if col not in exclude_columns]
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")
        
        # Fit and transform
        data_copy[columns_to_scale] = self.scaler.fit_transform(data_copy[columns_to_scale])
        
        print(f"Scaled {len(columns_to_scale)} features using {method} scaling")
        return data_copy
    
    def select_features(self, data, target_column='price', method='mutual_info', k=50):
        """
        Select the most important features
        """
        # Prepare features and target
        feature_columns = [col for col in data.columns if col not in ['date', target_column]]
        X = data[feature_columns].fillna(0)  # Fill NaN with 0 for feature selection
        y = data[target_column]
        
        # Remove rows where target is NaN
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        if method == 'mutual_info':
            self.feature_selector = SelectKBest(score_func=mutual_info_regression, k=min(k, len(feature_columns)))
        elif method == 'f_regression':
            self.feature_selector = SelectKBest(score_func=f_regression, k=min(k, len(feature_columns)))
        else:
            raise ValueError("Method must be 'mutual_info' or 'f_regression'")
        
        # Fit feature selector
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_mask = self.feature_selector.get_support()
        self.selected_features = [feature_columns[i] for i, selected in enumerate(selected_mask) if selected]
        
        print(f"Selected {len(self.selected_features)} features using {method}")
        
        # Return data with selected features plus date and target
        result_columns = ['date', target_column] + self.selected_features
        result_columns = [col for col in result_columns if col in data.columns]
        
        return data[result_columns]
    
    def apply_pca(self, data, target_column='price', n_components=0.95):
        """
        Apply Principal Component Analysis for dimensionality reduction
        """
        # Prepare features
        feature_columns = [col for col in data.columns if col not in ['date', target_column]]
        X = data[feature_columns].fillna(0)
        
        # Apply PCA
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X)
        
        # Create new dataframe with PCA components
        pca_columns = [f'pca_component_{i+1}' for i in range(X_pca.shape[1])]
        pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=data.index)
        
        # Combine with date and target
        result_data = pd.concat([data[['date', target_column]], pca_df], axis=1)
        
        print(f"Applied PCA: reduced {len(feature_columns)} features to {X_pca.shape[1]} components")
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        return result_data
    
    def engineer_all_features(self, data, target_column='price', use_pca=False, n_features=50):
        """
        Apply all feature engineering techniques
        """
        print("Starting comprehensive feature engineering...")
        
        # Create all types of features
        data = self.create_lag_features(data, target_column)
        data = self.create_rolling_features(data, target_column)
        data = self.create_momentum_features(data, target_column)
        data = self.create_volatility_features(data, target_column)
        data = self.create_price_position_features(data, target_column)
        data = self.create_trend_features(data, target_column)
        data = self.create_cyclical_features(data)
        data = self.create_interaction_features(data)
        
        # Remove rows with too many NaN values
        data = data.dropna(thresh=len(data.columns) * 0.7)  # Keep rows with at least 70% non-NaN values
        
        # Scale features
        data = self.scale_features(data)
        
        # Feature selection or PCA
        if use_pca:
            data = self.apply_pca(data, target_column)
        else:
            data = self.select_features(data, target_column, k=n_features)
        
        print(f"Feature engineering complete. Final dataset shape: {data.shape}")
        return data
    
    def get_feature_importance_scores(self):
        """
        Get feature importance scores from feature selection
        """
        if self.feature_selector is None:
            return None
        
        scores = self.feature_selector.scores_
        feature_names = [col for col in self.selected_features]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'score': scores[self.feature_selector.get_support()]
        }).sort_values('score', ascending=False)
        
        return importance_df