#!/usr/bin/env python3
"""
Live Chat API with Real-time Gold Price Integration
Supports multiple data sources and WebSocket connections
"""

from flask import Blueprint, request, jsonify, render_template, current_app
from flask_socketio import emit, join_room, leave_room
import json
import threading
import time
from datetime import datetime, timedelta
import requests
import yfinance as yf
from typing import Dict, List, Optional
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.real_time_fetcher import RealTimeGoldDataFetcher

live_chat_bp = Blueprint('live_chat', __name__)

class LiveGoldPriceChat:
    """
    Live chat system with integrated gold price updates from multiple sources
    """
    
    def __init__(self, socketio=None):
        self.socketio = socketio
        self.fetcher = RealTimeGoldDataFetcher()
        self.active_rooms = set()
        self.price_cache = {}
        self.update_interval = 30  # seconds
        self.is_running = False
        
        # Supported data sources with their characteristics
        self.data_sources = {
            'yahoo_finance': {
                'name': 'Yahoo Finance',
                'symbol': 'GC=F',
                'description': 'COMEX Gold Futures - Most reliable for real-time data',
                'update_frequency': '1 minute',
                'reliability': 'High',
                'features': ['Price', 'Volume', 'OHLC', 'Change%']
            },
            'metals_api': {
                'name': 'Metals-API',
                'description': 'Professional precious metals API',
                'update_frequency': '1 minute',
                'reliability': 'High',
                'features': ['Spot Price', 'Historical Data']
            },
            'alpha_vantage': {
                'name': 'Alpha Vantage',
                'description': 'Financial data provider with gold ETF tracking',
                'update_frequency': '5 minutes',
                'reliability': 'Medium',
                'features': ['ETF Price', 'Technical Indicators']
            },
            'finnhub': {
                'name': 'Finnhub',
                'description': 'Real-time financial data',
                'update_frequency': '1 minute',
                'reliability': 'Medium',
                'features': ['Real-time Price', 'Market Data']
            },
            'business_insider': {
                'name': 'Business Insider',
                'description': 'Financial news and market data',
                'current_price': 3447.94,
                'reliability': 'Medium',
                'features': ['Market Analysis', 'News Integration']
            },
            'apmex': {
                'name': 'APMEX',
                'description': 'Precious metals dealer pricing',
                'current_price': 3463.30,
                'reliability': 'High',
                'features': ['Dealer Price', 'Physical Gold']
            },
            'kitco': {
                'name': 'Kitco',
                'description': 'Leading precious metals information',
                'current_price': 3446.50,
                'reliability': 'High',
                'features': ['Spot Price', 'Market News']
            },
            'veracash': {
                'name': 'VeraCash',
                'description': 'Digital precious metals platform',
                'current_price': 3451.24,
                'reliability': 'Medium',
                'features': ['Digital Trading', 'Real-time Price']
            }
        }
        
    def start_price_updates(self):
        """Start background thread for price updates"""
        if not self.is_running:
            self.is_running = True
            thread = threading.Thread(target=self._price_update_loop)
            thread.daemon = True
            thread.start()
            
    def stop_price_updates(self):
        """Stop background price updates"""
        self.is_running = False
        
    def _price_update_loop(self):
        """Background loop for fetching and broadcasting price updates"""
        while self.is_running:
            try:
                # Fetch prices from all available sources
                price_data = self._fetch_all_sources()
                
                if price_data and self.socketio:
                    # Broadcast to all active rooms
                    for room in self.active_rooms:
                        self.socketio.emit('price_update', price_data, room=room)
                        
                # Cache the data
                self.price_cache = price_data
                
            except Exception as e:
                print(f"Error in price update loop: {e}")
                
            time.sleep(self.update_interval)
            
    def _fetch_all_sources(self) -> Dict:
        """Fetch prices from all available sources"""
        sources_data = {}
        
        # Fetch from API sources
        for source_key in ['yahoo', 'metals_api', 'alpha_vantage', 'finnhub']:
            try:
                data = self.fetcher.get_current_gold_price(source=source_key)
                if data:
                    sources_data[source_key] = {
                        'price': data.get('price', 0),
                        'change': data.get('change', 0),
                        'change_percent': data.get('change_percent', 0),
                        'timestamp': data.get('timestamp', datetime.now().isoformat()),
                        'source_info': self.data_sources.get(f'{source_key}_finance' if source_key == 'yahoo' else source_key, {})
                    }
            except Exception as e:
                print(f"Error fetching from {source_key}: {e}")
                
        # Add static sources with current prices
        static_sources = ['business_insider', 'apmex', 'kitco', 'veracash']
        for source in static_sources:
            if source in self.data_sources:
                sources_data[source] = {
                    'price': self.data_sources[source].get('current_price', 0),
                    'timestamp': datetime.now().isoformat(),
                    'source_info': self.data_sources[source]
                }
                
        # Calculate average and best/worst prices
        if sources_data:
            prices = [data['price'] for data in sources_data.values() if data['price'] > 0]
            if prices:
                sources_data['summary'] = {
                    'average_price': round(sum(prices) / len(prices), 2),
                    'highest_price': max(prices),
                    'lowest_price': min(prices),
                    'price_spread': round(max(prices) - min(prices), 2),
                    'total_sources': len(prices),
                    'timestamp': datetime.now().isoformat()
                }
                
        return sources_data
        
    def get_source_recommendations(self) -> Dict:
        """Get recommendations for best data sources for live chat integration"""
        recommendations = {
            'best_for_live_chat': {
                'primary': 'yahoo_finance',
                'reason': 'Most reliable real-time data with 1-minute updates',
                'backup': 'kitco',
                'backup_reason': 'Established precious metals authority'
            },
            'most_accurate': {
                'source': 'apmex',
                'reason': 'Dealer pricing reflects actual market conditions',
                'price': 3463.30
            },
            'fastest_updates': {
                'sources': ['yahoo_finance', 'metals_api', 'finnhub'],
                'update_frequency': '1 minute'
            },
            'integration_strategy': {
                'websocket_primary': 'yahoo_finance',
                'fallback_sources': ['kitco', 'apmex', 'metals_api'],
                'update_interval': '30 seconds',
                'error_handling': 'Automatic fallback to next available source'
            }
        }
        return recommendations

# Initialize the live chat system
live_chat_system = LiveGoldPriceChat()

@live_chat_bp.route('/chat')
def chat_interface():
    """Render the live chat interface"""
    return render_template('live_chat.html')

@live_chat_bp.route('/live-chat')
def live_chat_interface():
    """Alternative route for live chat interface"""
    return render_template('live_chat.html')

@live_chat_bp.route('/api/sources')
def get_data_sources():
    """Get information about all available data sources"""
    return jsonify({
        'sources': live_chat_system.data_sources,
        'recommendations': live_chat_system.get_source_recommendations(),
        'current_prices': live_chat_system.price_cache
    })

@live_chat_bp.route('/api/current-prices')
def get_current_prices():
    """Get current prices from all sources"""
    if not live_chat_system.price_cache:
        # Fetch fresh data if cache is empty
        live_chat_system.price_cache = live_chat_system._fetch_all_sources()
    
    return jsonify(live_chat_system.price_cache)

@live_chat_bp.route('/api/source-analysis')
def analyze_sources():
    """Analyze which sources are best for live chat integration"""
    analysis = {
        'recommended_integration': {
            'primary_source': {
                'name': 'Yahoo Finance (GC=F)',
                'reasons': [
                    'Real-time COMEX gold futures data',
                    '1-minute update frequency',
                    'High reliability and uptime',
                    'Includes volume and OHLC data',
                    'Free API access'
                ],
                'implementation': 'WebSocket connection for live updates'
            },
            'secondary_sources': [
                {
                    'name': 'Kitco',
                    'price': 3446.50,
                    'reason': 'Industry standard for spot gold prices'
                },
                {
                    'name': 'APMEX',
                    'price': 3463.30,
                    'reason': 'Reflects actual dealer pricing'
                }
            ]
        },
        'live_chat_features': {
            'real_time_updates': 'Every 30 seconds',
            'price_alerts': 'Configurable thresholds',
            'multi_source_comparison': 'Side-by-side pricing',
            'historical_tracking': 'Price movement over time',
            'prediction_integration': 'ML model predictions in chat'
        },
        'technical_implementation': {
            'websocket_support': True,
            'fallback_mechanism': 'Automatic source switching',
            'caching_strategy': '1-minute cache with real-time updates',
            'error_handling': 'Graceful degradation'
        }
    }
    
    return jsonify(analysis)

# WebSocket event handlers (requires Flask-SocketIO)
def init_socketio_handlers(socketio):
    """Initialize WebSocket event handlers"""
    live_chat_system.socketio = socketio
    
    @socketio.on('join_price_room')
    def on_join(data):
        room = data.get('room', 'general')
        join_room(room)
        live_chat_system.active_rooms.add(room)
        
        # Send current prices immediately
        if live_chat_system.price_cache:
            emit('price_update', live_chat_system.price_cache)
        
        # Start price updates if not already running
        live_chat_system.start_price_updates()
        
    @socketio.on('leave_price_room')
    def on_leave(data):
        room = data.get('room', 'general')
        leave_room(room)
        live_chat_system.active_rooms.discard(room)
        
    @socketio.on('request_price_update')
    def on_price_request():
        # Force immediate price update
        price_data = live_chat_system._fetch_all_sources()
        emit('price_update', price_data)
        
    @socketio.on('chat_message')
    def handle_message(data):
        # Handle chat messages with price context
        message = data.get('message', '')
        room = data.get('room', 'general')
        
        # Add current price context to messages about gold
        if any(keyword in message.lower() for keyword in ['gold', 'price', 'prediction']):
            current_summary = live_chat_system.price_cache.get('summary', {})
            if current_summary:
                data['price_context'] = {
                    'current_avg': current_summary.get('average_price'),
                    'timestamp': current_summary.get('timestamp')
                }
        
        # Broadcast message to room
        socketio.emit('chat_message', data, room=room)