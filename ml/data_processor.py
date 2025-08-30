import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """
    Data collection and preprocessing for gold price prediction
    """
    
    def __init__(self):
        self.raw_data = None
        self.processed_data = None
        self.external_factors = None
        
    def load_data_from_file(self, file_path=None):
        """
        Load data from CSV file with comprehensive error handling
        """
        try:
            if file_path is None:
                # Try to find the latest gold_prices file
                from data.data_collector import GoldDataCollector
                collector = GoldDataCollector()
                data = collector.load_latest_data('gold_prices')
                if data is not None:
                    self.raw_data = data
                    print(f"Loaded {len(data)} records from latest gold_prices file")
                    return data
                else:
                    print("No gold_prices files found, generating sample data...")
                    return self.generate_sample_data()
            else:
                import pandas as pd
                data = pd.read_csv(file_path)
                data['date'] = pd.to_datetime(data['date'])
                self.raw_data = data
                print(f"Loaded {len(data)} records from {file_path}")
                return data
                
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Falling back to sample data generation...")
            return self.generate_sample_data()
        
    def fetch_gold_prices(self, symbol='GOLD', period='2y'):
        """
        Fetch gold price data from Yahoo Finance
        GC=F is the symbol for Gold Futures
        """
        try:
            # Fetch gold price data
            gold_data = yf.download(symbol, period=period, interval='1d')
            
            # Reset index to make date a column
            gold_data.reset_index(inplace=True)
            
            # Rename columns
            gold_data.columns = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
            
            # Use adjusted close as the main price
            gold_data['price'] = gold_data['adj_close']
            
            # Select relevant columns
            gold_data = gold_data[['date', 'open', 'high', 'low', 'close', 'price', 'volume']]
            
            self.raw_data = gold_data
            print(f"Fetched {len(gold_data)} days of gold price data")
            
            return gold_data
            
        except Exception as e:
            print(f"Error fetching gold price data: {e}")
            return None
    
    def fetch_economic_indicators(self):
        """
        Fetch economic indicators that affect gold prices
        This is a placeholder - in production, you'd use APIs like FRED, Alpha Vantage, etc.
        """
        try:
            # Fetch USD Index (DXY)
            usd_data = yf.download('DX-Y.NYB', period='2y', interval='1d')
            usd_data.reset_index(inplace=True)
            usd_data = usd_data[['Date', 'Adj Close']]
            usd_data.columns = ['date', 'usd_index']
            
            # Fetch S&P 500 (as stock market indicator)
            sp500_data = yf.download('^GSPC', period='2y', interval='1d')
            sp500_data.reset_index(inplace=True)
            sp500_data = sp500_data[['Date', 'Adj Close']]
            sp500_data.columns = ['date', 'sp500']
            
            # Fetch 10-Year Treasury Yield
            treasury_data = yf.download('^TNX', period='2y', interval='1d')
            treasury_data.reset_index(inplace=True)
            treasury_data = treasury_data[['Date', 'Adj Close']]
            treasury_data.columns = ['date', 'treasury_yield']
            
            # Fetch Oil prices (crude oil)
            oil_data = yf.download('CL=F', period='2y', interval='1d')
            oil_data.reset_index(inplace=True)
            oil_data = oil_data[['Date', 'Adj Close']]
            oil_data.columns = ['date', 'oil_price']
            
            # Merge all economic indicators
            economic_data = usd_data
            for data in [sp500_data, treasury_data, oil_data]:
                economic_data = pd.merge(economic_data, data, on='date', how='outer')
            
            # Forward fill missing values
            economic_data = economic_data.fillna(method='ffill')
            
            self.external_factors = economic_data
            print(f"Fetched economic indicators for {len(economic_data)} days")
            
            return economic_data
            
        except Exception as e:
            print(f"Error fetching economic indicators: {e}")
            return None
    
    def generate_sample_data(self, days=500):
        """
        Generate sample gold price data for testing
        """
        np.random.seed(42)
        
        # Generate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate realistic gold price data with trend and seasonality
        base_price = 1800  # Base gold price
        trend = np.linspace(0, 200, len(dates))  # Upward trend
        seasonality = 50 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)  # Yearly seasonality
        noise = np.random.normal(0, 20, len(dates))  # Random noise
        
        prices = base_price + trend + seasonality + noise
        
        # Generate OHLC data
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            # Generate realistic OHLC values
            daily_volatility = np.random.normal(0, 15)
            high = price + abs(daily_volatility) + np.random.uniform(5, 15)
            low = price - abs(daily_volatility) - np.random.uniform(5, 15)
            open_price = price + np.random.uniform(-10, 10)
            close_price = price + np.random.uniform(-5, 5)
            volume = np.random.randint(10000, 100000)
            
            data.append({
                'date': date,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close_price, 2),
                'price': round(price, 2),
                'volume': volume
            })
        
        sample_data = pd.DataFrame(data)
        self.raw_data = sample_data
        
        print(f"Generated {len(sample_data)} days of sample gold price data")
        return sample_data
    
    def clean_data(self, data=None):
        """
        Clean and preprocess the data with comprehensive error handling
        """
        try:
            if data is None:
                if self.raw_data is not None:
                    data = self.raw_data.copy()
                else:
                    print("No raw data available, loading from file...")
                    data = self.load_data_from_file()
            
            if data is None or len(data) == 0:
                print("No data available for cleaning, generating sample data...")
                data = self.generate_sample_data()
            
            # Ensure required columns exist
            required_columns = ['date', 'price']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                print(f"Missing required columns: {missing_columns}")
                if 'close' in data.columns and 'price' not in data.columns:
                    data['price'] = data['close']
                    print("Used 'close' column as 'price'")
            
            # Remove duplicates
            initial_len = len(data)
            data = data.drop_duplicates(subset=['date'])
            if len(data) < initial_len:
                print(f"Removed {initial_len - len(data)} duplicate records")
            
            # Sort by date
            data = data.sort_values('date')
            
            # Handle missing values
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                # Use forward fill for missing values
                data[numeric_columns] = data[numeric_columns].fillna(method='ffill')
                # Use backward fill for any remaining missing values
                data[numeric_columns] = data[numeric_columns].fillna(method='bfill')
                # Fill any remaining with median
                for col in numeric_columns:
                    if data[col].isna().any():
                        data[col] = data[col].fillna(data[col].median())
            
            # Remove outliers (prices beyond 3 standard deviations) only if we have enough data
            if 'price' in data.columns and len(data) > 10:
                price_mean = data['price'].mean()
                price_std = data['price'].std()
                if price_std > 0:  # Avoid division by zero
                    lower_bound = price_mean - 3 * price_std
                    upper_bound = price_mean + 3 * price_std
                    
                    outliers = (data['price'] < lower_bound) | (data['price'] > upper_bound)
                    outlier_count = outliers.sum()
                    if outlier_count > 0:
                        print(f"Removing {outlier_count} outliers")
                        data = data[~outliers]
            
            # Reset index
            data = data.reset_index(drop=True)
            
            print(f"Cleaned data: {len(data)} records")
            return data
            
        except Exception as e:
            print(f"Error during data cleaning: {e}")
            print("Generating clean sample data as fallback...")
            return self.generate_sample_data()
    
    def add_technical_indicators(self, data=None):
        """
        Add technical indicators to the data with adaptive window sizes
        """
        if data is None:
            data = self.raw_data.copy()
        
        if data is None:
            raise ValueError("No data available. Please fetch or generate data first.")
        
        # Adaptive window sizes based on data length
        data_len = len(data)
        max_window = min(50, data_len // 3)  # Use at most 1/3 of data length
        
        # Moving averages with adaptive windows
        if data_len >= 5:
            data['sma_5'] = data['price'].rolling(window=5).mean()
        if data_len >= 10:
            data['sma_10'] = data['price'].rolling(window=10).mean()
        if data_len >= 20:
            data['sma_20'] = data['price'].rolling(window=20).mean()
        if max_window >= 20:
            data['sma_50'] = data['price'].rolling(window=min(max_window, 50)).mean()
        
        # Exponential moving averages with adaptive spans
        if data_len >= 12:
            data['ema_12'] = data['price'].ewm(span=min(12, max_window)).mean()
        if data_len >= 26:
            data['ema_26'] = data['price'].ewm(span=min(26, max_window)).mean()
        
        # MACD (only if we have both EMAs)
        if 'ema_12' in data.columns and 'ema_26' in data.columns:
            data['macd'] = data['ema_12'] - data['ema_26']
            data['macd_signal'] = data['macd'].ewm(span=9).mean()
        
        # RSI with adaptive window
        rsi_window = min(14, max_window)
        if data_len >= rsi_window:
            delta = data['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands with adaptive window
        bb_window = min(20, max_window)
        if data_len >= bb_window:
            data['bb_middle'] = data['price'].rolling(window=bb_window).mean()
            bb_std = data['price'].rolling(window=bb_window).std()
            data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
            data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        
        # Price changes with adaptive periods
        data['price_change'] = data['price'].pct_change()
        if data_len >= 5:
            data['price_change_5'] = data['price'].pct_change(5)
        if data_len >= 10:
            data['price_change_10'] = data['price'].pct_change(10)
        
        # Volatility with adaptive windows
        if data_len >= 5:
            data['volatility_5'] = data['price'].rolling(window=5).std()
        if data_len >= 10:
            data['volatility_10'] = data['price'].rolling(window=10).std()
        if data_len >= 20:
            data['volatility_20'] = data['price'].rolling(window=20).std()
        
        print("Added technical indicators")
        return data
    
    def merge_external_factors(self, gold_data=None, external_data=None):
        """
        Merge gold price data with external economic factors
        """
        if gold_data is None:
            gold_data = self.raw_data
        if external_data is None:
            external_data = self.external_factors
            
        if gold_data is None or external_data is None:
            print("Missing data for merging")
            return gold_data
        
        # Ensure date columns are datetime and normalize timezones
        gold_data['date'] = pd.to_datetime(gold_data['date']).dt.tz_localize(None)
        external_data['date'] = pd.to_datetime(external_data['date']).dt.tz_localize(None)
        
        # Merge data
        merged_data = pd.merge(gold_data, external_data, on='date', how='left')
        
        # Forward fill missing external factor values
        external_columns = external_data.columns.drop('date')
        merged_data[external_columns] = merged_data[external_columns].fillna(method='ffill')
        
        print(f"Merged data: {len(merged_data)} records with external factors")
        return merged_data
    
    def prepare_training_data(self, use_external_factors=True):
        """
        Prepare complete training dataset
        """
        if self.raw_data is None:
            print("No raw data available. Generating sample data...")
            self.generate_sample_data()
        
        # Clean data
        clean_data = self.clean_data()
        
        # Add technical indicators
        data_with_indicators = self.add_technical_indicators(clean_data)
        
        # Merge with external factors if available and requested
        if use_external_factors and self.external_factors is not None:
            final_data = self.merge_external_factors(data_with_indicators)
        else:
            final_data = data_with_indicators
        
        # Remove rows with NaN values
        final_data = final_data.dropna()
        
        self.processed_data = final_data
        
        print(f"Prepared training data: {len(final_data)} records")
        return final_data
    
    def get_latest_data(self, days=30):
        """
        Get the most recent data for prediction
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Please prepare training data first.")
        
        return self.processed_data.tail(days)
    
    def save_data(self, filename='gold_price_data.csv'):
        """
        Save processed data to CSV
        """
        if self.processed_data is not None:
            self.processed_data.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
        else:
            print("No processed data to save")
    
    def load_data(self, filename='gold_price_data.csv'):
        """
        Load data from CSV
        """
        try:
            self.processed_data = pd.read_csv(filename)
            self.processed_data['date'] = pd.to_datetime(self.processed_data['date'])
            print(f"Loaded data from {filename}: {len(self.processed_data)} records")
            return self.processed_data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None