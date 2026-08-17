#!/usr/bin/env python3
"""
SocketIO Event Handlers for Gold Price Live Chat
"""

from flask_socketio import emit, join_room, leave_room
from flask import current_app, request
import json
from datetime import datetime
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.real_time_fetcher import RealTimeGoldDataFetcher

def register_socketio_events(socketio):
    """Register all SocketIO event handlers"""
    
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection"""
        print(f"Client connected: {request.sid}")
        emit('status', {'message': 'Connected to Gold Price Live Chat'})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection"""
        print(f"Client disconnected: {request.sid}")
    
    @socketio.on('join_price_room')
    def handle_join_room(data):
        """Handle joining a price update room"""
        room = data.get('room', 'general')
        join_room(room)
        emit('status', {'message': f'Joined room: {room}'})
        print(f"Client {request.sid} joined room: {room}")
    
    @socketio.on('leave_price_room')
    def handle_leave_room(data):
        """Handle leaving a price update room"""
        room = data.get('room', 'general')
        leave_room(room)
        emit('status', {'message': f'Left room: {room}'})
        print(f"Client {request.sid} left room: {room}")
    
    @socketio.on('chat_message')
    def handle_chat_message(data):
        """Handle chat messages with price context"""
        try:
            message = data.get('message', '')
            room = data.get('room', 'general')
            user = data.get('user', 'Anonymous')
            
            # Add price context if available
            price_context = None
            try:
                fetcher = RealTimeGoldDataFetcher()
                current_prices = fetcher.get_all_sources_data()
                if current_prices and 'summary' in current_prices:
                    price_context = {
                        'current_avg': current_prices['summary'].get('average_price', 'N/A'),
                        'timestamp': datetime.now().isoformat()
                    }
            except Exception as e:
                print(f"Error getting price context: {e}")
            
            # Broadcast message to room
            message_data = {
                'user': user,
                'message': message,
                'timestamp': datetime.now().isoformat(),
                'price_context': price_context
            }
            
            socketio.emit('chat_message', message_data, room=room)
            print(f"Chat message from {user} in {room}: {message}")
            
        except Exception as e:
            print(f"Error handling chat message: {e}")
            emit('error', {'message': 'Failed to send message'})
    
    @socketio.on('request_price_update')
    def handle_price_update_request():
        """Handle manual price update requests"""
        try:
            fetcher = RealTimeGoldDataFetcher()
            price_data = fetcher.get_all_sources_data()
            
            if price_data:
                emit('price_update', price_data)
                print("Manual price update sent")
            else:
                emit('error', {'message': 'Failed to fetch price data'})
                
        except Exception as e:
            print(f"Error handling price update request: {e}")
            emit('error', {'message': 'Failed to get price update'})

def broadcast_price_update(socketio, price_data, room='general'):
    """Broadcast price updates to all clients in a room"""
    try:
        socketio.emit('price_update', price_data, room=room)
        print(f"Price update broadcasted to room: {room}")
    except Exception as e:
        print(f"Error broadcasting price update: {e}")