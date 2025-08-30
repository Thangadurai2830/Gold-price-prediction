import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import json
import os
from typing import Dict, Optional, List

class RealTimeGoldDataFetcher:
    """
    Real-time gold price data fetcher with multiple data sources
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 60  # Cache for 1 minute
        self.last_fetch_time = {}
        
        # API endpoints
        self.apis = {
            'yahoo': self._fetch_yahoo_finance,
            'metals_api': self._fetch_metals_api,
            'alpha_vantage': self._fetch_alpha_vantage,
            'finnhub': self._fetch_finnhub
        }
        
        # API keys (set these as environment variables)
        self.api_keys = {
            'metals_api': os.getenv('METALS_API_KEY'),
            'alpha_vantage': os.getenv('ALPHA_VANTAGE_KEY'),
            'finnhub': os.getenv('FINNHUB_KEY')
        }
    
    def get_current_gold_price(self, source='yahoo') -> Dict:
        """
        Get current gold price from specified source
        """
        cache_key = f"current_price_{source}"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        try:
            if source in self.apis:
                data = self.apis[source]()
                if data:
                    self.cache[cache_key] = data
                    self.last_fetch_time[cache_key] = datetime.now()
                    return data
            
            # Fallback to other sources
            for fallback_source in ['yahoo', 'metals_api', 'alpha_vantage']:
                if fallback_source != source and fallback_source in self.apis:
                    try:
                        data = self.apis[fallback_source]()
                        if data:
                            self.cache[cache_key] = data
                            self.last_fetch_time[cache_key] = datetime.now()
                            return data
                    except:
                        continue
            
            # If all sources fail, return cached data or default
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            return self._get_default_price()
            
        except Exception as e:
            print(f"Error fetching gold price from {source}: {e}")
            return self._get_default_price()
    
    def _fetch_yahoo_finance(self) -> Dict:
        """
        Fetch gold price from Yahoo Finance
        """
        try:
            # Gold futures symbol
            ticker = yf.Ticker("GC=F")
            
            # Get current data
            info = ticker.info
            hist = ticker.history(period="1d", interval="1m")
            
            if not hist.empty:
                latest = hist.iloc[-1]
                current_price = float(latest['Close'])
                
                # Calculate changes
                prev_close = float(info.get('previousClose', current_price))
                change = current_price - prev_close
                change_percent = (change / prev_close) * 100 if prev_close > 0 else 0
                
                return {
                    'price': round(current_price, 2),
                    'change': round(change, 2),
                    'change_percent': round(change_percent, 2),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'yahoo_finance',
                    'volume': int(latest.get('Volume', 0)),
                    'high': round(float(latest['High']), 2),
                    'low': round(float(latest['Low']), 2),
                    'open': round(float(latest['Open']), 2)
                }
        except Exception as e:
            print(f"Yahoo Finance error: {e}")
            return None
    
    def _fetch_metals_api(self) -> Dict:
        """
        Fetch gold price from Metals-API (requires API key)
        """
        if not self.api_keys.get('metals_api'):
            return None
            
        try:
            url = f"https://metals-api.com/api/latest?access_key={self.api_keys['metals_api']}&base=USD&symbols=XAU"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success') and 'rates' in data:
                    # Metals API returns price per gram, convert to per ounce
                    price_per_gram = data['rates'].get('XAU', 0)
                    price_per_ounce = price_per_gram * 31.1035 if price_per_gram > 0 else 0
                    
                    return {
                        'price': round(price_per_ounce, 2),
                        'timestamp': datetime.now().isoformat(),
                        'source': 'metals_api'
                    }
        except Exception as e:
            print(f"Metals API error: {e}")
            return None
    
    def _fetch_alpha_vantage(self) -> Dict:
        """
        Fetch gold price from Alpha Vantage (requires API key)
        """
        if not self.api_keys.get('alpha_vantage'):
            return None
            
        try:
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=GLD&apikey={self.api_keys['alpha_vantage']}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                quote = data.get('Global Quote', {})
                
                if quote:
                    price = float(quote.get('05. price', 0))
                    change = float(quote.get('09. change', 0))
                    change_percent = quote.get('10. change percent', '0%').replace('%', '')
                    
                    return {
                        'price': round(price, 2),
                        'change': round(change, 2),
                        'change_percent': round(float(change_percent), 2),
                        'timestamp': datetime.now().isoformat(),
                        'source': 'alpha_vantage'
                    }
        except Exception as e:
            print(f"Alpha Vantage error: {e}")
            return None
    
    def _fetch_finnhub(self) -> Dict:
        """
        Fetch gold price from Finnhub (requires API key)
        """
        if not self.api_keys.get('finnhub'):
            return None
            
        try:
            url = f"https://finnhub.io/api/v1/quote?symbol=GLD&token={self.api_keys['finnhub']}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'c' in data:  # Current price
                    current_price = float(data['c'])
                    prev_close = float(data.get('pc', current_price))
                    change = current_price - prev_close
                    change_percent = (change / prev_close) * 100 if prev_close > 0 else 0
                    
                    return {
                        'price': round(current_price, 2),
                        'change': round(change, 2),
                        'change_percent': round(change_percent, 2),
                        'high': round(float(data.get('h', current_price)), 2),
                        'low': round(float(data.get('l', current_price)), 2),
                        'open': round(float(data.get('o', current_price)), 2),
                        'timestamp': datetime.now().isoformat(),
                        'source': 'finnhub'
                    }
        except Exception as e:
            print(f"Finnhub error: {e}")
            return None
    
    def get_historical_data(self, days=30) -> pd.DataFrame:
        """
        Get historical gold price data
        """
        try:
            ticker = yf.Ticker("GC=F")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            data = ticker.history(start=start_date, end=end_date)
            
            if not data.empty:
                data.reset_index(inplace=True)
                data['price'] = data['Close']
                data['date'] = data['Date']
                
                return data[['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'price']]
            
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            
        return pd.DataFrame()
    
    def get_market_indicators(self) -> Dict:
        """
        Get related market indicators that affect gold prices
        """
        indicators = {}
        
        try:
            # US Dollar Index
            usd_ticker = yf.Ticker("DX-Y.NYB")
            usd_data = usd_ticker.history(period="1d")
            if not usd_data.empty:
                indicators['usd_index'] = round(float(usd_data.iloc[-1]['Close']), 2)
            
            # 10-Year Treasury Yield
            treasury_ticker = yf.Ticker("^TNX")
            treasury_data = treasury_ticker.history(period="1d")
            if not treasury_data.empty:
                indicators['treasury_10y'] = round(float(treasury_data.iloc[-1]['Close']), 2)
            
            # VIX (Volatility Index)
            vix_ticker = yf.Ticker("^VIX")
            vix_data = vix_ticker.history(period="1d")
            if not vix_data.empty:
                indicators['vix'] = round(float(vix_data.iloc[-1]['Close']), 2)
            
            # Oil Price
            oil_ticker = yf.Ticker("CL=F")
            oil_data = oil_ticker.history(period="1d")
            if not oil_data.empty:
                indicators['oil_wti'] = round(float(oil_data.iloc[-1]['Close']), 2)
            
        except Exception as e:
            print(f"Error fetching market indicators: {e}")
        
        return indicators
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """
        Check if cached data is still valid
        """
        if cache_key not in self.cache or cache_key not in self.last_fetch_time:
            return False
        
        time_diff = datetime.now() - self.last_fetch_time[cache_key]
        return time_diff.total_seconds() < self.cache_duration
    
    def _get_default_price(self) -> Dict:
        """
        Return default price data when all sources fail
        """
        return {
            'price': 2000.00,
            'change': 0.00,
            'change_percent': 0.00,
            'timestamp': datetime.now().isoformat(),
            'source': 'fallback',
            'status': 'using_fallback_data'
        }
    
    def get_comprehensive_data(self) -> Dict:
        """
        Get comprehensive gold market data
        """
        current_price = self.get_current_gold_price()
        market_indicators = self.get_market_indicators()
        
        return {
            'gold_price': current_price,
            'market_indicators': market_indicators,
            'timestamp': datetime.now().isoformat()
        }

# Global instance
real_time_fetcher = RealTimeGoldDataFetcher()