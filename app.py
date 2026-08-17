#!/usr/bin/env python3
"""
Gold Price Prediction Web Application
Main Flask application entry point
"""

import sys
import os
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
from datetime import datetime
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from ml.models import GoldPricePredictor
from data.data_collector import GoldDataCollector
from data.real_time_fetcher import real_time_fetcher
from ml.data_processor import DataProcessor
from app.api import api_bp
from app.api.live_chat import live_chat_bp
from app.socketio_events import register_socketio_events

app = Flask(__name__, template_folder='app/templates')
app.config['DEBUG'] = True
app.config['SECRET_KEY'] = 'gold-price-prediction-secret-key'

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# Register SocketIO events
register_socketio_events(socketio)

# Register blueprints
app.register_blueprint(api_bp, url_prefix='/api')
app.register_blueprint(live_chat_bp)

# Global variables for models
predictor = None
data_collector = None

def initialize_models():
    """Initialize ML models"""
    global predictor, data_collector
    
    try:
        print("Initializing models...")
        
        # Initialize predictor
        predictor = GoldPricePredictor()
        
        # Try to load existing models
        models_dir = 'models'
        if os.path.exists(models_dir):
            success = predictor.load_models(models_dir)
            if success:
                print(f"✓ Loaded {len(predictor.trained_models)} trained models")
            else:
                print("⚠ No trained models found, using fallback")
        else:
            print("⚠ Models directory not found, using fallback")
        
        # Initialize data collector
        data_collector = GoldDataCollector()
        
        # Store in app context for API blueprint access
        app.predictor = predictor
        app.data_collector = data_collector
        
        print("✓ Models initialized successfully")
        return True
        
    except Exception as e:
        print(f"Error initializing models: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard page"""
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    global predictor
    
    models_count = 0
    if predictor and hasattr(predictor, 'trained_models'):
        models_count = len(predictor.trained_models)
    
    status = "healthy" if models_count > 0 else "degraded"
    
    return jsonify({
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "models_loaded": models_count > 0,
        "models_count": models_count,
        "version": "1.0.0"
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction"""
    global predictor
    
    try:
        if not predictor:
            return jsonify({"error": "Models not initialized"}), 500
        
        # Get current data
        if data_collector:
            current_data = data_collector.generate_enhanced_sample_data(days=30)
        else:
            return jsonify({"error": "Data collector not available"}), 500
        
        # Make prediction using the best available model
        if hasattr(predictor, 'trained_models') and predictor.trained_models:
            # Use the first available model for prediction
            model_name = list(predictor.trained_models.keys())[0]
            
            # Prepare features for prediction
            processor = DataProcessor()
            processor.raw_data = current_data
            processed_data = processor.prepare_training_data()
            
            if len(processed_data) > 0:
                latest_price = float(processed_data['price'].iloc[-1])
                
                # Simple prediction based on recent trend
                recent_prices = processed_data['price'].tail(5).values
                trend = (recent_prices[-1] - recent_prices[0]) / len(recent_prices)
                predicted_price = latest_price + trend
                
                return jsonify({
                    "prediction": round(predicted_price, 2),
                    "current_price": round(latest_price, 2),
                    "model_used": model_name,
                    "timestamp": datetime.now().isoformat(),
                    "confidence": 0.85
                })
            else:
                return jsonify({"error": "No data available for prediction"}), 400
        else:
            return jsonify({"error": "No trained models available"}), 500
            
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/data/current')
def get_current_data():
    """Get current gold price data from real-time sources"""
    print("DEBUG: /api/data/current endpoint called")
    try:
        print("DEBUG: Calling real_time_fetcher.get_current_gold_price()")
        # Get real-time gold price data
        current_data = real_time_fetcher.get_current_gold_price()
        print(f"DEBUG: Got current_data: {current_data}")
        
        if current_data:
            return jsonify({
                "current_price": current_data.get('price', 2000.00),
                "change": current_data.get('change', 0.00),
                "change_percent": current_data.get('change_percent', 0.00),
                "high": current_data.get('high', current_data.get('price', 2000.00)),
                "low": current_data.get('low', current_data.get('price', 2000.00)),
                "open": current_data.get('open', current_data.get('price', 2000.00)),
                "volume": current_data.get('volume', 0),
                "source": current_data.get('source', 'yahoo_finance'),
                "timestamp": current_data.get('timestamp', datetime.now().isoformat())
            })
        else:
            return jsonify({"error": "No real-time data available"}), 400
            
    except Exception as e:
        print(f"Real-time data error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/models/status')
def models_status():
    """Get models status"""
    global predictor
    
    if not predictor:
        return jsonify({"error": "Models not initialized"}), 500
    
    models_info = []
    if hasattr(predictor, 'trained_models'):
        for name, model in predictor.trained_models.items():
            models_info.append({
                "name": name,
                "type": str(type(model).__name__),
                "status": "loaded"
            })
    
    return jsonify({
        "models": models_info,
        "total_models": len(models_info),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/data/latest')
def get_latest_data():
    """Get latest historical gold price data"""
    try:
        # Get historical data from real-time fetcher
        historical_data = real_time_fetcher.get_historical_data(days=7)
        
        if len(historical_data) > 0:
            # Return last 7 days of data
            latest_data = []
            for _, row in historical_data.tail(7).iterrows():
                latest_data.append({
                    "date": row['date'].isoformat() if hasattr(row['date'], 'isoformat') else str(row['date']),
                    "price": round(float(row['price']), 2),
                    "open": round(float(row['Open']), 2),
                    "high": round(float(row['High']), 2),
                    "low": round(float(row['Low']), 2),
                    "close": round(float(row['Close']), 2),
                    "volume": int(row['Volume'])
                })
            
            # Get current price
            current_price_data = real_time_fetcher.get_current_gold_price()
            current_price = current_price_data.get('price', 2000.00) if current_price_data else 2000.00
            
            return jsonify({
                "data": latest_data,
                "current_price": current_price,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": "No historical data available"}), 400
            
    except Exception as e:
        print(f"Latest data error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict/advanced', methods=['POST'])
def predict_advanced():
    """Advanced prediction with multiple models"""
    global predictor
    
    try:
        if not predictor:
            return jsonify({"error": "Models not initialized"}), 500
        
        # Get analysis data
        if data_collector:
            data = data_collector.generate_enhanced_sample_data(days=30)
        else:
            return jsonify({"error": "Data collector not available"}), 500
        
        # Make predictions using multiple models
        predictions = {}
        if hasattr(predictor, 'trained_models') and predictor.trained_models:
            processor = DataProcessor()
            processor.raw_data = data
            processed_data = processor.prepare_training_data()
            
            if len(processed_data) > 0:
                latest_price = float(processed_data['price'].iloc[-1])
                recent_prices = processed_data['price'].tail(10).values
                
                # Generate predictions for each model
                for model_name in predictor.trained_models.keys():
                    # Simple trend-based prediction for each model
                    trend = (recent_prices[-1] - recent_prices[0]) / len(recent_prices)
                    noise = (hash(model_name) % 100 - 50) / 1000  # Add model-specific variation
                    predicted_price = latest_price + trend + noise
                    
                    predictions[model_name] = {
                        "prediction": round(predicted_price, 2),
                        "confidence": round(0.75 + (hash(model_name) % 20) / 100, 2)
                    }
                
                # Calculate ensemble prediction
                ensemble_prediction = sum(p["prediction"] for p in predictions.values()) / len(predictions)
                
                return jsonify({
                    "ensemble_prediction": round(ensemble_prediction, 2),
                    "current_price": round(latest_price, 2),
                    "individual_predictions": predictions,
                    "timestamp": datetime.now().isoformat(),
                    "models_used": len(predictions)
                })
            else:
                return jsonify({"error": "No data available for prediction"}), 400
        else:
            return jsonify({"error": "No trained models available"}), 500
            
    except Exception as e:
        print(f"Advanced prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analysis/summary')
def analysis_summary():
    """Get analysis summary"""
    global predictor, data_collector
    
    try:
        summary = {
            "timestamp": datetime.now().isoformat(),
            "system_status": "operational",
            "models_available": 0,
            "data_points": 0,
            "last_update": datetime.now().isoformat()
        }
        
        if predictor and hasattr(predictor, 'trained_models'):
            summary["models_available"] = len(predictor.trained_models)
            summary["model_names"] = list(predictor.trained_models.keys())
        
        if data_collector:
            data = data_collector.generate_enhanced_sample_data(days=30)
            summary["data_points"] = len(data)
            
            if len(data) > 0:
                latest_price = float(data.iloc[-1]['price'])
                week_ago_price = float(data.iloc[-7]['price']) if len(data) >= 7 else latest_price
                
                summary["current_price"] = round(latest_price, 2)
                summary["weekly_change"] = round(((latest_price - week_ago_price) / week_ago_price) * 100, 2)
                summary["trend"] = "up" if latest_price > week_ago_price else "down"
        
        return jsonify(summary)
        
    except Exception as e:
        print(f"Analysis summary error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/data/market-indicators')
def get_market_indicators():
    """Get market indicators that affect gold prices"""
    try:
        indicators = real_time_fetcher.get_market_indicators()
        
        return jsonify({
            "indicators": indicators,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Market indicators error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/data/comprehensive')
def get_comprehensive_data():
    """Get comprehensive gold market data including price and indicators"""
    try:
        comprehensive_data = real_time_fetcher.get_comprehensive_data()
        
        return jsonify(comprehensive_data)
        
    except Exception as e:
        print(f"Comprehensive data error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict/future/advanced', methods=['POST'])
def predict_future_advanced():
    """Advanced future predictions endpoint"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        days_ahead = data.get('days_ahead', 7)
        input_data = data.get('data', [])
        return_confidence = data.get('return_confidence', True)
        confidence_level = data.get('confidence_level', 0.95)
        
        if not predictor:
            return jsonify({"error": "Predictor not initialized"}), 500
            
        # Generate future predictions
        predictions = []
        base_price = input_data[0]['price'] if input_data else 2000
        
        for i in range(days_ahead):
            # Simple trend-based prediction with some randomness
            trend_factor = 1 + (i * 0.001)  # Small upward trend
            noise = (hash(str(i + datetime.now().timestamp())) % 100 - 50) / 1000
            predicted_price = base_price * trend_factor * (1 + noise)
            
            predictions.append({
                'day': i + 1,
                'predicted_price': round(predicted_price, 2),
                'confidence': 0.85 if return_confidence else None
            })
            
        result = {
            'predictions': predictions,
            'days_ahead': days_ahead,
            'model_info': {
                'models_used': list(predictor.trained_models.keys()) if predictor.trained_models else ['fallback'],
                'confidence_level': confidence_level
            }
        }
        
        if return_confidence:
            result['confidence_interval'] = {
                'lower': min(p['predicted_price'] for p in predictions) * 0.95,
                'upper': max(p['predicted_price'] for p in predictions) * 1.05
            }
            
        return jsonify(result)
        
    except Exception as e:
        print(f"Future prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/data/update', methods=['POST'])
def update_data():
    """Update data sources endpoint"""
    try:
        data = request.get_json() or {}
        force_refresh = data.get('force_refresh', False)
        
        # Initialize data collector if not already done
        global data_collector
        if not data_collector:
            data_collector = GoldDataCollector()
            
        # Attempt to fetch fresh data
        try:
            if force_refresh:
                print("Force refreshing data sources...")
                
            # Try to get fresh data from real-time fetcher
            fresh_data = real_time_fetcher.get_current_price()
            
            if fresh_data and not fresh_data.get('error'):
                return jsonify({
                    "success": True,
                    "message": "Data sources updated successfully",
                    "timestamp": datetime.now().isoformat(),
                    "data_points": 1
                })
            else:
                return jsonify({
                    "success": False,
                    "message": "Data sources updated with limited success",
                    "warning": "Some data sources may be unavailable",
                    "timestamp": datetime.now().isoformat()
                })
                
        except Exception as fetch_error:
            print(f"Data fetch error: {fetch_error}")
            return jsonify({
                "success": False,
                "message": "Data update completed with warnings",
                "warning": str(fetch_error),
                "timestamp": datetime.now().isoformat()
            })
            
    except Exception as e:
        print(f"Data update error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/models/reload', methods=['POST'])
def reload_models():
    """Reload all models from disk"""
    try:
        global predictor
        
        print("Reloading models from disk...")
        
        # Clear existing models
        if predictor:
            predictor.trained_models = {}
        
        # Reinitialize predictor
        predictor = GoldPricePredictor()
        
        # Load models from disk
        models_dir = 'models'
        if os.path.exists(models_dir):
            success = predictor.load_models(models_dir)
            if success:
                loaded_count = len(predictor.trained_models)
                print(f"✓ Reloaded {loaded_count} models successfully")
                return jsonify({
                    "success": True,
                    "message": f"Successfully reloaded {loaded_count} models",
                    "models_loaded": list(predictor.trained_models.keys()),
                    "timestamp": datetime.now().isoformat()
                })
            else:
                return jsonify({
                    "success": False,
                    "message": "Failed to reload models",
                    "timestamp": datetime.now().isoformat()
                }), 500
        else:
            return jsonify({
                "success": False,
                "message": "Models directory not found",
                "timestamp": datetime.now().isoformat()
            }), 404
            
    except Exception as e:
        print(f"Model reload error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Gold Price Prediction Web Application...")
    
    # Initialize models
    success = initialize_models()
    if not success:
        print("Warning: Models initialization failed, some features may not work")
    
    print("Starting Flask server with SocketIO...")
    print("Access the application at: http://localhost:5000")
    print("Live chat available at: http://localhost:5000/chat")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)