import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
import json
import time
import os
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class GoldDataCollector:
    """
    Comprehensive data collector for gold prices and related economic indicators
    """
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.ensure_data_directory()
        
    def ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'raw'), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'processed'), exist_ok=True)
        
    def fetch_gold_prices_yahoo(self, symbol='GC=F', period='2y', interval='1d'):
        """
        Fetch gold price data from Yahoo Finance
        
        Args:
            symbol: Gold futures symbol (GC=F for COMEX Gold)
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        """
        try:
            print(f"Fetching gold price data for {symbol}...")
            
            # Download data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                print("No data received from Yahoo Finance")
                return None
                
            # Reset index to make date a column
            data.reset_index(inplace=True)
            
            # Handle timezone-aware datetime
            if 'Date' in data.columns and hasattr(data['Date'].dtype, 'tz') and data['Date'].dtype.tz is not None:
                data['Date'] = data['Date'].dt.tz_convert('UTC').dt.tz_localize(None)
            elif 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
            
            # Standardize column names based on actual columns
            if 'Adj Close' in data.columns:
                data = data.rename(columns={
                    'Date': 'date',
                    'Open': 'open',
                    'High': 'high', 
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume',
                    'Adj Close': 'adj_close'
                })
                data['price'] = data['adj_close']
            else:
                # Handle case where columns might be different
                data.columns = [col.lower().replace(' ', '_') for col in data.columns]
                if 'close' in data.columns:
                    data['price'] = data['close']
                elif 'adj_close' in data.columns:
                    data['price'] = data['adj_close']
            
            # Select only needed columns
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'price']
            available_cols = [col for col in required_cols if col in data.columns]
            data = data[available_cols]
            
            print(f"Successfully fetched {len(data)} records")
            
            # Save raw data
            filename = f"gold_prices_{symbol.replace('=', '_')}_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = os.path.join(self.data_dir, 'raw', filename)
            data.to_csv(filepath, index=False)
            print(f"Data saved to {filepath}")
            
            return data
            
        except Exception as e:
            print(f"Error fetching gold price data: {e}")
            return None
    
    def fetch_economic_indicators(self, period='2y'):
        """
        Fetch economic indicators that influence gold prices
        """
        indicators = {
            'USD_Index': 'DX-Y.NYB',  # US Dollar Index
            'SP500': '^GSPC',         # S&P 500
            'Treasury_10Y': '^TNX',   # 10-Year Treasury Yield
            'Oil_WTI': 'CL=F',        # WTI Crude Oil
            'VIX': '^VIX',            # Volatility Index
            'Silver': 'SI=F',         # Silver Futures
            'Copper': 'HG=F',         # Copper Futures
            'NASDAQ': '^IXIC',        # NASDAQ Composite
            'EUR_USD': 'EURUSD=X',    # EUR/USD Exchange Rate
        }
        
        all_data = {}
        
        for name, symbol in indicators.items():
            try:
                print(f"Fetching {name} ({symbol})...")
                
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval='1d')
                
                if not data.empty:
                    data.reset_index(inplace=True)
                    data = data[['Date', 'Close']]
                    data.columns = ['date', f'{name.lower()}']
                    all_data[name] = data
                    
                    # Small delay to avoid rate limiting
                    time.sleep(0.1)
                else:
                    print(f"No data for {name}")
                    
            except Exception as e:
                print(f"Error fetching {name}: {e}")
                continue
        
        if all_data:
            # Merge all indicators
            merged_data = None
            for name, data in all_data.items():
                if merged_data is None:
                    merged_data = data
                else:
                    merged_data = pd.merge(merged_data, data, on='date', how='outer')
            
            # Sort by date and forward fill missing values
            merged_data = merged_data.sort_values('date')
            merged_data = merged_data.fillna(method='ffill')
            
            # Save economic indicators
            filename = f"economic_indicators_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = os.path.join(self.data_dir, 'raw', filename)
            merged_data.to_csv(filepath, index=False)
            print(f"Economic indicators saved to {filepath}")
            
            return merged_data
        
        return None
    
    def fetch_alternative_gold_data(self):
        """
        Fetch gold price data from alternative sources (placeholder for APIs like Alpha Vantage, etc.)
        """
        # This is a placeholder for additional data sources
        # In production, you might use APIs like:
        # - Alpha Vantage
        # - Quandl
        # - FRED (Federal Reserve Economic Data)
        # - Metals-API
        
        print("Alternative data sources not implemented yet")
        return None
    
    def fetch_news_sentiment(self):
        """
        Fetch news sentiment data related to gold (placeholder)
        """
        # This would integrate with news APIs to get sentiment scores
        # Examples: NewsAPI, Finnhub, Alpha Vantage News
        
        print("News sentiment data not implemented yet")
        return None
    
    def generate_sample_data(self, start_date=None, end_date=None, frequency='D'):
        """
        Generate realistic sample gold price data for testing
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=730)  # 2 years ago
        if end_date is None:
            end_date = datetime.now()
            
        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq=frequency)
        
        # Generate realistic gold price data
        np.random.seed(42)
        
        # Base parameters
        base_price = 1800
        trend_strength = 0.1
        volatility = 0.02
        seasonal_amplitude = 50
        
        # Generate price series
        prices = []
        current_price = base_price
        
        for i, date in enumerate(dates):
            # Trend component
            trend = trend_strength * i / len(dates) * 100
            
            # Seasonal component (yearly cycle)
            seasonal = seasonal_amplitude * np.sin(2 * np.pi * i / 365.25)
            
            # Random walk component
            random_change = np.random.normal(0, volatility * current_price)
            
            # Market regime changes (occasional jumps)
            if np.random.random() < 0.01:  # 1% chance of regime change
                random_change += np.random.normal(0, volatility * current_price * 5)
            
            # Update price
            current_price = base_price + trend + seasonal + random_change
            prices.append(max(current_price, 100))  # Ensure price doesn't go below $100
        
        # Generate OHLCV data
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            # Generate realistic OHLC values
            daily_range = np.random.uniform(0.005, 0.03) * price  # 0.5% to 3% daily range
            
            high = price + np.random.uniform(0.3, 0.7) * daily_range
            low = price - np.random.uniform(0.3, 0.7) * daily_range
            
            open_price = low + np.random.uniform(0, 1) * (high - low)
            close_price = low + np.random.uniform(0, 1) * (high - low)
            
            # Volume (higher volume on larger price moves)
            price_change = abs(close_price - open_price) / open_price
            base_volume = np.random.randint(50000, 200000)
            volume = int(base_volume * (1 + price_change * 10))
            
            data.append({
                'date': date,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close_price, 2),
                'price': round(close_price, 2),
                'volume': volume
            })
        
        sample_data = pd.DataFrame(data)
        
        # Save sample data
        filename = f"gold_prices_sample_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = os.path.join(self.data_dir, 'raw', filename)
        sample_data.to_csv(filepath, index=False)
        print(f"Generated {len(sample_data)} sample records, saved to {filepath}")
        
        return sample_data
    
    def generate_enhanced_sample_data(self, days=2000):
        """
        Generate enhanced sample data with more realistic patterns and external factors
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Generate date range (business days only)
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        n_days = len(dates)
        
        np.random.seed(42)
        
        # Enhanced gold price modeling
        base_price = 1800
        
        # Multiple trend components
        long_trend = np.linspace(0, 400, n_days)  # Long-term upward trend
        medium_trend = 50 * np.sin(2 * np.pi * np.arange(n_days) / 252)  # Annual cycle
        short_trend = 20 * np.sin(2 * np.pi * np.arange(n_days) / 63)  # Quarterly cycle
        
        # Market regime changes
        regime_changes = np.random.choice([0, 1], n_days, p=[0.95, 0.05])
        regime_effect = np.cumsum(regime_changes * np.random.normal(0, 30, n_days))
        
        # Volatility clustering (GARCH-like)
        volatility = np.zeros(n_days)
        volatility[0] = 15
        for i in range(1, n_days):
            volatility[i] = 0.1 + 0.85 * volatility[i-1] + 0.1 * (np.random.normal(0, 1) ** 2)
        
        # Price innovations
        innovations = np.random.normal(0, 1, n_days) * np.sqrt(volatility)
        
        # Combine all components
        log_prices = np.log(base_price) + (long_trend + medium_trend + short_trend + regime_effect) / base_price
        log_prices += np.cumsum(innovations) / 100
        
        prices = np.exp(log_prices)
        
        # Generate OHLC data
        daily_returns = np.diff(np.log(prices))
        daily_returns = np.concatenate([[0], daily_returns])
        
        # High/Low based on intraday volatility
        intraday_vol = volatility * 0.3
        high_prices = prices * np.exp(np.abs(np.random.normal(0, intraday_vol)))
        low_prices = prices * np.exp(-np.abs(np.random.normal(0, intraday_vol)))
        
        # Open prices (gap from previous close)
        open_prices = np.zeros_like(prices)
        open_prices[0] = prices[0]
        for i in range(1, len(prices)):
            gap = np.random.normal(0, volatility[i] * 0.1)
            open_prices[i] = prices[i-1] * np.exp(gap)
        
        # Volume with realistic patterns
        base_volume = 50000
        volume_trend = np.random.normal(1, 0.3, n_days)
        volume_volatility = 1 + 0.5 * np.abs(daily_returns) * 10  # Higher volume on big moves
        volumes = base_volume * volume_trend * volume_volatility
        volumes = np.maximum(volumes, 1000).astype(int)
        
        # Create enhanced DataFrame
        data = pd.DataFrame({
            'date': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': prices,
            'price': prices,  # Main price column
            'volume': volumes,
            'volatility': volatility,
            'returns': daily_returns
        })
        
        # Ensure OHLC consistency
        data['high'] = data[['open', 'high', 'low', 'close']].max(axis=1)
        data['low'] = data[['open', 'high', 'low', 'close']].min(axis=1)
        
        # Save enhanced sample data
        filename = f"gold_prices_enhanced_sample_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = os.path.join(self.data_dir, 'raw', filename)
        data.to_csv(filepath, index=False)
        print(f"Generated {len(data)} enhanced sample records, saved to {filepath}")
        
        return data
    
    def load_latest_data(self, data_type='gold_prices'):
        """
        Load the most recent data file of specified type
        """
        raw_dir = os.path.join(self.data_dir, 'raw')
        
        if not os.path.exists(raw_dir):
            print(f"Raw data directory {raw_dir} does not exist")
            return None
        
        # Find files matching the data type
        files = [f for f in os.listdir(raw_dir) if f.startswith(data_type) and f.endswith('.csv')]
        
        if not files:
            print(f"No {data_type} files found in {raw_dir}")
            return None
        
        # Get the most recent file
        latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(raw_dir, x)))
        filepath = os.path.join(raw_dir, latest_file)
        
        try:
            data = pd.read_csv(filepath)
            data['date'] = pd.to_datetime(data['date'])
            print(f"Loaded {len(data)} records from {latest_file}")
            return data
        except Exception as e:
            print(f"Error loading {latest_file}: {e}")
            return None
    
    def update_data(self, force_refresh=False):
        """
        Update all data sources
        """
        print("Starting data update process...")
        
        # Check if we need to update (daily update)
        last_update_file = os.path.join(self.data_dir, 'last_update.txt')
        
        if not force_refresh and os.path.exists(last_update_file):
            with open(last_update_file, 'r') as f:
                last_update = datetime.fromisoformat(f.read().strip())
            
            if (datetime.now() - last_update).days < 1:
                print("Data is up to date (updated within 24 hours)")
                return
        
        # Fetch gold prices
        gold_data = self.fetch_gold_prices_yahoo()
        
        # If real data fetch fails, generate sample data
        if gold_data is None:
            print("Real data fetch failed, generating sample data...")
            gold_data = self.generate_sample_data()
        
        # Fetch economic indicators
        economic_data = self.fetch_economic_indicators()
        
        # Update last update timestamp
        with open(last_update_file, 'w') as f:
            f.write(datetime.now().isoformat())
        
        print("Data update completed")
        
        return {
            'gold_data': gold_data,
            'economic_data': economic_data
        }
    
    def get_data_summary(self):
        """
        Get summary of available data
        """
        raw_dir = os.path.join(self.data_dir, 'raw')
        processed_dir = os.path.join(self.data_dir, 'processed')
        
        summary = {
            'raw_files': [],
            'processed_files': [],
            'total_raw_size': 0,
            'total_processed_size': 0
        }
        
        # Check raw files
        if os.path.exists(raw_dir):
            for file in os.listdir(raw_dir):
                if file.endswith('.csv'):
                    filepath = os.path.join(raw_dir, file)
                    size = os.path.getsize(filepath)
                    summary['raw_files'].append({
                        'name': file,
                        'size_mb': round(size / (1024*1024), 2),
                        'modified': datetime.fromtimestamp(os.path.getmtime(filepath))
                    })
                    summary['total_raw_size'] += size
        
        # Check processed files
        if os.path.exists(processed_dir):
            for file in os.listdir(processed_dir):
                if file.endswith('.csv'):
                    filepath = os.path.join(processed_dir, file)
                    size = os.path.getsize(filepath)
                    summary['processed_files'].append({
                        'name': file,
                        'size_mb': round(size / (1024*1024), 2),
                        'modified': datetime.fromtimestamp(os.path.getmtime(filepath))
                    })
                    summary['total_processed_size'] += size
        
        summary['total_raw_size_mb'] = round(summary['total_raw_size'] / (1024*1024), 2)
        summary['total_processed_size_mb'] = round(summary['total_processed_size'] / (1024*1024), 2)
        
        return summary
    
    def clean_old_data(self, days_to_keep=30):
        """
        Clean up old data files
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for directory in [os.path.join(self.data_dir, 'raw'), os.path.join(self.data_dir, 'processed')]:
            if not os.path.exists(directory):
                continue
                
            files_removed = 0
            for file in os.listdir(directory):
                filepath = os.path.join(directory, file)
                if os.path.getmtime(filepath) < cutoff_date.timestamp():
                    os.remove(filepath)
                    files_removed += 1
            
            print(f"Removed {files_removed} old files from {directory}")

# Example usage and testing
if __name__ == "__main__":
    collector = GoldDataCollector()
    
    # Generate sample data for testing
    sample_data = collector.generate_sample_data()
    
    # Get data summary
    summary = collector.get_data_summary()
    print("\nData Summary:")
    print(f"Raw files: {len(summary['raw_files'])}")
    print(f"Total raw data size: {summary['total_raw_size_mb']} MB")