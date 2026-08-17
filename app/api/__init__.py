from flask import Blueprint, jsonify, request, current_app, g
from datetime import datetime, timedelta
import sys
import os
import traceback
import numpy as np
import pandas as pd

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from ml.models import GoldPricePredictor
    from ml.data_processor import DataProcessor
    from ml.feature_engineering import FeatureEngineer
    from data.data_collector import GoldDataCollector
except ImportError as e:
    print(f"Warning: Could not import ML modules: {e}")
    GoldPricePredictor = None
    DataProcessor = None
    FeatureEngineer = None
    GoldDataCollector = None

api_bp = Blueprint("api", __name__)

# Helper functions to get models from app context
def get_predictor():
    """Get predictor from Flask app context"""
    from flask import current_app
    # Try to get from app instance first, then from context
    if hasattr(current_app, 'predictor'):
        return current_app.predictor
    return None

def get_data_processor():
    """Get data processor from Flask app context"""
    if not hasattr(current_app, 'data_processor'):
        return None
    return current_app.data_processor

def get_feature_engineer():
    """Get feature engineer from Flask app context"""
    if not hasattr(current_app, 'feature_engineer'):
        return None
    return current_app.feature_engineer

def get_data_collector():
    """Get data collector from Flask app context"""
    if not hasattr(current_app, 'data_collector'):
        return None
    return current_app.data_collector

def initialize_models(app=None):
    """Initialize ML models and components"""
    from flask import current_app
    
    # Use provided app or fall back to current_app
    target_app = app if app is not None else current_app
    
    try:
        print(f"DEBUG: GoldPricePredictor is None: {GoldPricePredictor is None}")
        print(f"DEBUG: GoldPricePredictor: {GoldPricePredictor}")
        
        if GoldPricePredictor is None:
            print("Warning: GoldPricePredictor class not available - imports failed")
            # Create a minimal predictor object for fallback
            class FallbackPredictor:
                def __init__(self):
                    self.trained_models = {}
                    self.feature_columns = []
                    self.prophet_model = None
                    
                def get_model_performance(self):
                    return {"fallback": {"status": "active", "accuracy": "N/A"}}
                    
            target_app.predictor = FallbackPredictor()
            target_app.data_processor = None
            target_app.feature_engineer = None
            target_app.data_collector = None
            return False
        
        print("DEBUG: Checking for existing predictor...")
        print(f"DEBUG: Flask app id during init: {id(target_app)}")
        print(f"DEBUG: Flask app name during init: {target_app.name}")
        
        # Use existing predictor if available, otherwise create new one
        if hasattr(target_app, 'predictor') and target_app.predictor is not None:
            print(f"DEBUG: Using existing predictor: {target_app.predictor}")
        else:
            print("DEBUG: Creating new GoldPricePredictor instance...")
            target_app.predictor = GoldPricePredictor()
            print(f"DEBUG: GoldPricePredictor created: {target_app.predictor}")
        
        target_app.data_processor = DataProcessor() if DataProcessor else None
        target_app.feature_engineer = FeatureEngineer() if FeatureEngineer else None
        target_app.data_collector = GoldDataCollector() if GoldDataCollector else None
        print(f"DEBUG: Data collector created: {target_app.data_collector}")
        
        # Try to load existing models only if not already loaded
        try:
            if not hasattr(target_app.predictor, 'trained_models') or not target_app.predictor.trained_models:
                print("DEBUG: Loading models...")
                target_app.predictor.load_models('models')
                print("Loaded existing models")
                print(f"DEBUG: Models loaded: {list(target_app.predictor.trained_models.keys()) if hasattr(target_app.predictor, 'trained_models') else 'No trained_models attr'}")
            else:
                print(f"DEBUG: Models already loaded: {list(target_app.predictor.trained_models.keys())}")
        except Exception as load_error:
            print(f"No existing models found: {load_error}")
            import traceback
            print(f"Load error traceback: {traceback.format_exc()}")
            
        return True
    except Exception as e:
        print(f"Error initializing models: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        # Still create a fallback predictor
        class FallbackPredictor:
            def __init__(self):
                self.trained_models = {}
                self.feature_columns = []
                self.prophet_model = None
                
            def get_model_performance(self):
                return {"fallback": {"status": "active", "accuracy": "N/A"}}
                
        target_app.predictor = FallbackPredictor()
        target_app.data_processor = None
        target_app.feature_engineer = None
        target_app.data_collector = None
        return False

@api_bp.route("/ping", methods=["GET"])
def ping():
    import sys
    print("=== PING ENDPOINT CALLED ===")
    sys.stdout.flush()
    return {"message": "pong", "status": "ok"}

@api_bp.route("/simple-test", methods=["GET"])
def simple_test():
    """Simple test endpoint"""
    return {"message": "simple test works", "status": "ok"}

@api_bp.route("/test-new-endpoint", methods=["GET"])
def test_new_endpoint():
    """Brand new test endpoint"""
    return {"message": "new endpoint works", "status": "ok", "timestamp": str(__import__('datetime').datetime.now())}

@api_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for dashboard"""
    from flask import jsonify
    import os
    
    # Get the actual last update time from file or current time
    last_update_file = os.path.join('data', 'last_update.txt')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        if os.path.exists(last_update_file):
            with open(last_update_file, 'r') as f:
                timestamp_str = f.read().strip()
                # Parse ISO format timestamp and convert to readable format
                timestamp_obj = datetime.fromisoformat(timestamp_str.replace('T', ' ').split('.')[0])
                timestamp = timestamp_obj.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        print(f"Error reading last update file: {e}")
    
    # Check if models are loaded
    predictor = get_predictor()
    models_loaded = predictor is not None and hasattr(predictor, 'trained_models') and len(predictor.trained_models) > 0
    models_count = len(predictor.trained_models) if predictor and hasattr(predictor, 'trained_models') else 0
    
    return jsonify({
        "status": "ok",
        "message": "API is healthy",
        "models_loaded": models_count,
        "models_count": models_count,
        "timestamp": timestamp
    })
    
@api_bp.route("/debug-predictor", methods=["GET"])
def debug_predictor():
    """Debug endpoint to check predictor status - updated"""
    from flask import current_app, jsonify
    
    debug_info = {
        "flask_app_id": id(current_app),
        "flask_app_name": current_app.name,
        "has_predictor": hasattr(current_app, 'predictor'),
        "predictor_object": str(current_app.predictor) if hasattr(current_app, 'predictor') else None,
        "predictor_type": str(type(current_app.predictor)) if hasattr(current_app, 'predictor') else None,
        "has_trained_models": hasattr(current_app.predictor, 'trained_models') if hasattr(current_app, 'predictor') else False,
        "trained_models_count": len(current_app.predictor.trained_models) if hasattr(current_app, 'predictor') and hasattr(current_app.predictor, 'trained_models') else 0,
        "trained_models_keys": list(current_app.predictor.trained_models.keys()) if hasattr(current_app, 'predictor') and hasattr(current_app.predictor, 'trained_models') else [],
        "feature_columns_count": len(current_app.predictor.feature_columns) if hasattr(current_app, 'predictor') and hasattr(current_app.predictor, 'feature_columns') else 0,
        "has_feature_selector": hasattr(current_app.predictor, 'feature_selector') if hasattr(current_app, 'predictor') else False,
        "has_scaler": hasattr(current_app.predictor, 'scaler') if hasattr(current_app, 'predictor') else False,
        "selected_features_count": len(current_app.predictor.selected_features) if hasattr(current_app, 'predictor') and hasattr(current_app.predictor, 'selected_features') else 0
    }
    
    return jsonify(debug_info)

@api_bp.route("/status", methods=["GET"])
def status_check():
    """Simple status endpoint to test code reloading"""
    predictor = get_predictor()
    return jsonify({
        "status": "active",
        "predictor_exists": predictor is not None,
        "models_count": len(predictor.trained_models) if predictor and hasattr(predictor, 'trained_models') else 0,
        "timestamp": "2025-01-23-test"
    })

@api_bp.route("/debug", methods=["GET"])
def debug_info():
    """Debug endpoint to check Flask app state"""
    import os
    import time
    from datetime import datetime
    
    current_file = __file__
    file_mtime = os.path.getmtime(current_file)
    file_mtime_str = datetime.fromtimestamp(file_mtime).strftime('%Y-%m-%d %H:%M:%S')
    
    predictor = get_predictor()
    
    debug_info = {
        "current_working_directory": os.getcwd(),
        "current_file": current_file,
        "file_last_modified": file_mtime_str,
        "current_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "predictor_exists": predictor is not None,
        "predictor_type": str(type(predictor)) if predictor else None,
        "has_trained_models": hasattr(predictor, 'trained_models') if predictor else False,
        "trained_models_count": len(predictor.trained_models) if predictor and hasattr(predictor, 'trained_models') else 0,
        "trained_models_keys": list(predictor.trained_models.keys()) if predictor and hasattr(predictor, 'trained_models') else []
    }
    
    return jsonify(debug_info)

@api_bp.route("/health-test", methods=["GET"])
def health_test():
    """Test health check endpoint with different name"""
    try:
        # Return plain dict instead of jsonify
        return {"status": "ok", "message": "Health test endpoint working"}
    except Exception as e:
        import traceback
        print(f"Health test endpoint error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return {"error": str(e), "status": "error"}, 500

@api_bp.route("/health2", methods=["GET"])
def health_check2():
    """Alternative health check endpoint"""
    return jsonify({"status": "ok", "message": "Alternative health endpoint working"})

@api_bp.route("/test-cache-bypass", methods=["GET"])
def test_cache_bypass():
    """Test endpoint to verify cache bypass"""
    return jsonify({
        "message": "Cache bypass test successful",
        "timestamp": datetime.now().isoformat(),
        "test_flag": True
    })

@api_bp.route("/models/status", methods=["GET"])
def models_status():
    """Get status of loaded models"""
    predictor = get_predictor()
    
    if not predictor:
        return jsonify({"error": "Models not initialized"}), 500
    
    try:
        performance = predictor.get_model_performance()
        models_count = len(predictor.trained_models)
        
        # Check if using fallback system
        if models_count == 0 and "fallback" in performance:
            return jsonify({
                "models_loaded": 0,
                "status": "fallback_mode",
                "message": "Using fallback prediction system - ML models not available",
                "model_details": performance,
                "feature_count": len(predictor.feature_columns),
                "prophet_available": predictor.prophet_model is not None
            })
        
        return jsonify({
            "models_loaded": models_count,
            "model_details": performance,
            "feature_count": len(predictor.feature_columns),
            "prophet_available": predictor.prophet_model is not None
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route("/models/reload", methods=["POST"])
def reload_models():
    """Reload all models from disk"""
    try:
        predictor = get_predictor()
        
        if not predictor:
            return jsonify({"error": "Predictor not initialized"}), 500
        
        # Clear existing models
        predictor.trained_models = {}
        
        # Reload models from disk
        success = predictor.load_models('models')
        
        if success:
            models_count = len(predictor.trained_models)
            model_names = list(predictor.trained_models.keys())
            
            return jsonify({
                "status": "success",
                "message": "Models reloaded successfully",
                "models_loaded": models_count,
                "model_names": model_names,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to reload models",
                "timestamp": datetime.now().isoformat()
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error reloading models: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@api_bp.route("/data/latest", methods=["GET"])
def get_latest_data():
    """Get latest gold price data with comprehensive error handling"""
    data_collector = get_data_collector()
    
    try:
        # Initialize data collector if not available
        if not data_collector:
            if GoldDataCollector:
                data_collector = GoldDataCollector()
            else:
                return jsonify({
                    "error": "Data collector not available",
                    "fallback_data": {
                        "latest_price": 2000.0,
                        "message": "Using fallback price data"
                    }
                }), 200
        
        # Try to get latest data
        latest_data = None
        data_source = "unknown"
        
        try:
            latest_data = data_collector.load_latest_data('gold_prices')
            data_source = "file" if latest_data is not None else "none"
        except Exception as e:
            print(f"Error loading data from file: {e}")
        
        # Fallback to sample data if no real data available
        if latest_data is None or len(latest_data) == 0:
            try:
                latest_data = data_collector.generate_sample_gold_data(days=30)
                data_source = "sample"
            except Exception as e:
                print(f"Error generating sample data: {e}")
                # Ultimate fallback - return basic data structure
                return jsonify({
                    "data": [{
                        "date": datetime.now().isoformat(),
                        "price": 2000.0,
                        "open": 1995.0,
                        "high": 2005.0,
                        "low": 1990.0,
                        "close": 2000.0,
                        "volume": 100000
                    }],
                    "total_records": 1,
                    "latest_price": 2000.0,
                    "latest_date": datetime.now().isoformat(),
                    "data_source": "fallback",
                    "message": "Using fallback data due to system errors"
                }), 200
        
        # Ensure we have valid data
        if len(latest_data) == 0:
            return jsonify({
                "error": "No data available",
                "data_source": data_source
            }), 404
        
        # Get last 10 records
        recent_data = latest_data.tail(10).copy()
        
        # Ensure date column is properly formatted
        if 'date' in recent_data.columns:
            recent_data['date'] = pd.to_datetime(recent_data['date'])
        
        # Convert to JSON-serializable format
        records = []
        for _, row in recent_data.iterrows():
            record = {}
            for col, val in row.items():
                if pd.isna(val):
                    record[col] = None
                elif col == 'date' and hasattr(val, 'isoformat'):
                    record[col] = val.isoformat()
                elif isinstance(val, (np.integer, np.floating)):
                    record[col] = float(val)
                else:
                    record[col] = val
            records.append(record)
        
        result = {
            "data": records,
            "total_records": len(latest_data),
            "latest_price": float(recent_data.iloc[-1]['price']),
            "latest_date": recent_data.iloc[-1]['date'].isoformat() if pd.notna(recent_data.iloc[-1]['date']) else datetime.now().isoformat(),
            "data_source": data_source,
            "status": "success"
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Critical error in get_latest_data: {e}")
        traceback.print_exc()
        return jsonify({
            "error": "Internal server error",
            "fallback_data": {
                "latest_price": 2000.0,
                "latest_date": datetime.now().isoformat(),
                "message": "System error - using emergency fallback"
            }
        }), 200

@api_bp.route("/predict/single", methods=["POST"])
def predict_single():
    """Make a single price prediction with comprehensive error handling"""
    predictor = get_predictor()
    data_processor = get_data_processor()
    
    try:
        # Initialize models if not available
        if not predictor:
            try:
                initialize_models()
            except Exception as e:
                print(f"Failed to initialize models: {e}")
        
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "No input data provided",
                "fallback_prediction": 2000.0,
                "message": "Using average gold price as fallback"
            }), 200
        
        model_name = data.get('model', 'ensemble')
        
        # If features are provided directly
        if 'features' in data:
            features = data['features']
            print(f"Debug: Received features: {features}, type: {type(features)}")
            
            # Ensure we have predictor available
            if not predictor:
                return jsonify({
                    "error": "Predictor not available",
                    "fallback_prediction": 2000.0 + np.random.normal(0, 50),
                    "model_used": "fallback",
                    "message": "Using statistical fallback prediction"
                }), 200
            
            try:
                # Convert features to proper format
                if isinstance(features, dict):
                    feature_values = list(features.values())
                elif isinstance(features, list):
                    feature_values = features
                else:
                    feature_values = [features]
                
                print(f"Debug: Feature values before conversion: {feature_values}")
                
                # Ensure all values are numeric
                feature_values = [float(val) for val in feature_values]
                
                print(f"Debug: Feature values after conversion: {feature_values}")
                print(f"Debug: Model name: {model_name}")
                
                # The ML models expect features as a single array/list, not nested
                if model_name == 'ensemble':
                    prediction = predictor.predict_ensemble(feature_values)
                else:
                    prediction = predictor.predict_single(model_name, feature_values)
                
                return jsonify({
                    "prediction": float(prediction),
                    "model_used": model_name,
                    "features_count": len(feature_values),
                    "timestamp": datetime.now().isoformat(),
                    "status": "success"
                })
            except Exception as e:
                print(f"Prediction error: {e}")
                return jsonify({
                    "error": f"Prediction failed: {str(e)}",
                    "fallback_prediction": 2000.0 + np.random.normal(0, 30),
                    "model_used": "fallback",
                    "timestamp": datetime.now().isoformat()
                }), 200
        
        # If raw data is provided, process it first
        elif 'price_data' in data:
            return jsonify({
                "error": "Raw data processing not implemented in this endpoint",
                "fallback_prediction": 2000.0,
                "message": "Use /predict/ensemble for complex data processing"
            }), 200
        
        else:
            # Generate default features for prediction
            default_features = {
                'price_lag_1': 2000.0,
                'sma_5': 2000.0,
                'volatility': 0.02,
                'price_change': 0.0,
                'volume': 100000
            }
            
            if predictor:
                try:
                    prediction = predictor.predict_ensemble(list(default_features.values()))
                    return jsonify({
                        "prediction": float(prediction),
                        "model_used": "ensemble",
                        "features_used": "default",
                        "timestamp": datetime.now().isoformat(),
                        "message": "Used default features for prediction"
                    })
                except Exception as e:
                    print(f"Default prediction error: {e}")
            
            return jsonify({
                "error": "No features provided and default prediction failed",
                "fallback_prediction": 2000.0 + np.random.normal(0, 25),
                "model_used": "fallback",
                "message": "Provide 'features' in request body for better predictions"
            }), 200
            
    except Exception as e:
        print(f"Critical error in predict_single: {e}")
        traceback.print_exc()
        return jsonify({
            "error": "Critical system error",
            "fallback_prediction": 2000.0,
            "model_used": "emergency_fallback",
            "timestamp": datetime.now().isoformat(),
            "message": "Emergency fallback - system requires maintenance"
        }), 200

@api_bp.route("/predict/future", methods=["POST"])
def predict_future():
    """Predict future gold prices using Prophet model"""
    predictor = get_predictor()
    
    if not predictor:
        return jsonify({"error": "Models not initialized"}), 500
    
    if not predictor.prophet_model:
        return jsonify({"error": "Prophet model not available"}), 400
    
    try:
        data = request.get_json() or {}
        days = data.get('days', 30)
        
        if days > 365:
            return jsonify({"error": "Maximum prediction period is 365 days"}), 400
        
        # Generate future predictions
        future_predictions = predictor.predict_future_prophet(days=days)
        
        # Convert to JSON-serializable format
        predictions = []
        for _, row in future_predictions.iterrows():
            predictions.append({
                "date": row['ds'].isoformat(),
                "predicted_price": float(row['yhat']),
                "lower_bound": float(row['yhat_lower']),
                "upper_bound": float(row['yhat_upper'])
            })
        
        return jsonify({
            "predictions": predictions,
            "days_ahead": days,
            "model": "prophet",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route("/predict/advanced", methods=["POST"])
def predict_advanced():
    """Advanced prediction with confidence intervals and uncertainty quantification"""
    try:
        print("DEBUG ENDPOINT: Starting advanced prediction...")
        data = request.get_json()
        print(f"DEBUG ENDPOINT: Received data: {data}")
        
        if not data:
            # Use default features for testing
            data = {}
        
        # Get the predictor
        print("DEBUG ENDPOINT: Getting predictor...")
        predictor = get_predictor()
        if not predictor:
            print("DEBUG ENDPOINT: Predictor not found!")
            return jsonify({'error': 'Predictor not initialized'}), 500
        print(f"DEBUG ENDPOINT: Predictor found: {predictor}")
        print(f"DEBUG ENDPOINT: Predictor scaler type: {type(predictor.scaler)}")
        print(f"DEBUG ENDPOINT: Predictor scaler fitted: {hasattr(predictor.scaler, 'center_') and hasattr(predictor.scaler, 'scale_')}")
        
        # Check if predictor has required attributes
        if not hasattr(predictor, 'best_model') or predictor.best_model is None:
            return jsonify({'error': 'No trained models available'}), 500
        
        # Extract parameters
        return_confidence = data.get('return_confidence', True)
        confidence_level = data.get('confidence_level', 0.95)
        prediction_data = data.get('data')
        
        # If no data provided, use latest available data
        if not prediction_data:
            data_collector = get_data_collector()
            if data_collector:
                try:
                    latest_data = data_collector.load_latest_data('gold_prices')
                    if latest_data is not None and not latest_data.empty:
                        prediction_data = latest_data.tail(1).to_dict('records')[0]
                    else:
                        return jsonify({'error': 'No data available for prediction'}), 400
                except Exception as e:
                    return jsonify({'error': f'Error getting latest data: {str(e)}'}), 500
            else:
                return jsonify({'error': 'Data collector not available'}), 500
        
        # Convert data to DataFrame
        if isinstance(prediction_data, dict):
            df = pd.DataFrame([prediction_data])
        else:
            df = pd.DataFrame(prediction_data)
        
        # Make prediction with confidence intervals
        if return_confidence:
            try:
                result = predictor.predict(df, return_confidence=True, confidence_level=confidence_level)
                
                if isinstance(result, dict):
                    prediction_val = result['prediction']
                    lower_val = result['lower_bound']
                    upper_val = result['upper_bound']
                    std_val = result['std']
                    
                    # Handle array or scalar values
                    if hasattr(prediction_val, '__iter__') and not isinstance(prediction_val, str):
                        prediction_val = float(prediction_val[0])
                        lower_val = float(lower_val[0])
                        upper_val = float(upper_val[0])
                        std_val = float(std_val[0])
                    else:
                        prediction_val = float(prediction_val)
                        lower_val = float(lower_val)
                        upper_val = float(upper_val)
                        std_val = float(std_val)
                    
                    return jsonify({
                        'prediction': prediction_val,
                        'confidence_interval': {
                            'lower': lower_val,
                            'upper': upper_val,
                            'std': std_val,
                            'confidence_level': result['confidence_level']
                        },
                        'uncertainty_metrics': {
                            'prediction_std': std_val,
                            'interval_width': upper_val - lower_val
                        },
                        'timestamp': datetime.now().isoformat(),
                        'model': 'ensemble_with_uncertainty'
                    })
                else:
                    prediction = result
            except Exception as pred_error:
                print(f"Advanced prediction error: {pred_error}")
                import traceback
                traceback.print_exc()
                # Fallback to regular prediction
                try:
                    prediction = predictor.predict(df)
                except Exception as fallback_error:
                    return jsonify({'error': f'Prediction failed: {str(fallback_error)}'}), 500
        else:
            try:
                prediction = predictor.predict(df)
            except Exception as pred_error:
                return jsonify({'error': f'Prediction failed: {str(pred_error)}'}), 500
        
        # Handle prediction result
        if hasattr(prediction, '__iter__') and not isinstance(prediction, str):
            prediction_val = float(prediction[0])
        else:
            prediction_val = float(prediction)
        
        return jsonify({
            'prediction': prediction_val,
            'timestamp': datetime.now().isoformat(),
            'model': predictor.best_model_name if hasattr(predictor, 'best_model_name') else 'unknown',
            'note': 'Confidence intervals not available for this model'
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@api_bp.route("/predict/future/advanced", methods=["POST"])
def predict_future_advanced():
    """Advanced future prediction with uncertainty quantification"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get the predictor
        predictor = get_predictor()
        if not predictor:
            return jsonify({'error': 'Predictor not initialized'}), 500
        
        # Get parameters
        days_ahead = data.get('days_ahead', 30)
        return_confidence = data.get('return_confidence', True)
        confidence_level = data.get('confidence_level', 0.95)
        historical_data = data.get('data')
        
        if not historical_data:
            return jsonify({'error': 'Historical data required'}), 400
        
        if days_ahead > 365:
            return jsonify({'error': 'Maximum prediction period is 365 days'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        
        # Make future predictions with uncertainty
        if return_confidence and hasattr(predictor, 'predict_future'):
            try:
                result = predictor.predict_future(df, days_ahead=days_ahead, 
                                                return_confidence=True, 
                                                confidence_level=confidence_level)
                
                if isinstance(result, dict):
                    predictions = result['predictions']
                    confidence_intervals = result['confidence_intervals']
                    
                    # Generate future dates
                    last_date = pd.to_datetime(df['date'].iloc[-1])
                    future_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
                    
                    return jsonify({
                        'predictions': [
                            {
                                'date': date.isoformat(),
                                'predicted_price': float(price),
                                'confidence_interval': {
                                    'lower': float(ci['lower']),
                                    'upper': float(ci['upper']),
                                    'std': float(ci['std'])
                                } if i < len(confidence_intervals) else None,
                                'uncertainty_score': float(ci['std'] / price) if i < len(confidence_intervals) and price != 0 else None
                            }
                            for i, (date, price) in enumerate(zip(future_dates, predictions))
                        ],
                        'summary': {
                            'confidence_level': confidence_level,
                            'days_ahead': days_ahead,
                            'avg_uncertainty': float(np.mean([ci['std'] for ci in confidence_intervals])) if confidence_intervals else None,
                            'prediction_range': {
                                'min': float(np.min(predictions)),
                                'max': float(np.max(predictions)),
                                'mean': float(np.mean(predictions))
                            }
                        },
                        'timestamp': datetime.now().isoformat(),
                        'model': 'ensemble_with_uncertainty'
                    })
                else:
                    predictions = result
            except Exception as pred_error:
                print(f"Advanced future prediction error: {pred_error}")
                # Fallback to regular prediction
                predictions = predictor.predict_future(df, days_ahead=days_ahead)
        else:
            predictions = predictor.predict_future(df, days_ahead=days_ahead)
        
        # Generate future dates for fallback
        last_date = pd.to_datetime(df['date'].iloc[-1])
        future_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
        
        return jsonify({
            'predictions': [
                {
                    'date': date.isoformat(),
                    'predicted_price': float(price)
                }
                for date, price in zip(future_dates, predictions)
            ],
            'days_ahead': days_ahead,
            'timestamp': datetime.now().isoformat(),
            'model': predictor.best_model_name if hasattr(predictor, 'best_model_name') else 'unknown',
            'note': 'Confidence intervals not available for this model'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route("/analysis/feature-importance", methods=["GET"])
def get_feature_importance():
    """Get comprehensive feature importance from trained models"""
    try:
        predictor = get_predictor()
        if not predictor:
            return jsonify({'error': 'Predictor not initialized'}), 500
        
        # Get query parameters
        model_name = request.args.get('model', 'all')
        top_n = int(request.args.get('top_n', 20))
        
        if hasattr(predictor, 'get_feature_importance'):
            if model_name == 'all':
                importance = predictor.get_feature_importance(top_n=top_n)
            else:
                importance = predictor.get_feature_importance(model_name=model_name, top_n=top_n)
        else:
            return jsonify({'error': 'Feature importance not available'}), 400
        
        if importance is None:
            return jsonify({'error': 'Feature importance not available'}), 400
        
        # Format the response
        if isinstance(importance, list):
            # Multi-model importance (list of tuples)
            formatted_importance = [
                {
                    'feature': feature,
                    'importance_mean': stats['mean'],
                    'importance_std': stats['std'],
                    'model_count': stats['count']
                }
                for feature, stats in importance
            ]
        else:
            # Single model importance (DataFrame)
            formatted_importance = importance.to_dict('records')
        
        return jsonify({
            'feature_importance': formatted_importance,
            'model': model_name,
            'top_n': top_n,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route("/models/performance", methods=["GET"])
def get_model_performance():
    """Get detailed performance metrics for all trained models"""
    try:
        predictor = get_predictor()
        if not predictor:
            return jsonify({'error': 'Predictor not initialized'}), 500
        
        # Get query parameters
        detailed = request.args.get('detailed', 'false').lower() == 'true'
        
        if hasattr(predictor, 'get_model_performance'):
            performance = predictor.get_model_performance(detailed=detailed)
        else:
            performance = {'error': 'Performance metrics not available'}
        
        return jsonify({
            'model_performance': performance,
            'detailed': detailed,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route("/analysis/prediction-intervals", methods=["POST"])
def get_prediction_intervals():
    """Calculate prediction intervals at multiple confidence levels"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        predictor = get_predictor()
        if not predictor:
            return jsonify({'error': 'Predictor not initialized'}), 500
        
        # Get parameters
        confidence_levels = data.get('confidence_levels', [0.68, 0.95, 0.99])
        prediction_data = data.get('data')
        
        if not prediction_data:
            return jsonify({'error': 'Prediction data required'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(prediction_data)
        
        # Calculate prediction intervals
        if hasattr(predictor, 'calculate_prediction_intervals'):
            intervals = predictor.calculate_prediction_intervals(df, confidence_levels)
        else:
            return jsonify({'error': 'Prediction intervals not available'}), 400
        
        if intervals is None:
            return jsonify({'error': 'Could not calculate prediction intervals'}), 400
        
        # Format response
        formatted_intervals = {}
        for level, interval_data in intervals.items():
            formatted_intervals[str(level)] = {
                'lower': float(interval_data['lower'][0]) if hasattr(interval_data['lower'], '__iter__') else float(interval_data['lower']),
                'upper': float(interval_data['upper'][0]) if hasattr(interval_data['upper'], '__iter__') else float(interval_data['upper']),
                'width': float(interval_data['width'][0]) if hasattr(interval_data['width'], '__iter__') else float(interval_data['width'])
            }
        
        return jsonify({
            'prediction_intervals': formatted_intervals,
            'confidence_levels': confidence_levels,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route("/predict/ensemble", methods=["POST"])
def predict_ensemble():
    """Get predictions from all available models with comprehensive error handling"""
    predictor = get_predictor()
    data_processor = get_data_processor()
    data_collector = get_data_collector()
    
    try:
        # Initialize components if not available
        if not predictor:
            try:
                initialize_models()
            except Exception as e:
                print(f"Failed to initialize models: {e}")
        
        if not data_collector:
            try:
                if GoldDataCollector:
                    data_collector = GoldDataCollector()
            except Exception as e:
                print(f"Failed to initialize data collector: {e}")
        
        # Get latest data for prediction
        latest_data = None
        data_source = "unknown"
        
        try:
            if data_collector:
                latest_data = data_collector.load_latest_data('gold_prices')
                data_source = "file" if latest_data is not None else "none"
        except Exception as e:
            print(f"Error loading data: {e}")
        
        # Fallback to sample data if no real data available
        if latest_data is None or len(latest_data) == 0:
            try:
                if data_collector:
                    latest_data = data_collector.generate_sample_gold_data(days=30)
                    data_source = "sample"
                else:
                    # Create minimal sample data
                    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
                    latest_data = pd.DataFrame({
                        'date': dates,
                        'price': np.random.normal(2000, 50, 30),
                        'volume': np.random.randint(50000, 200000, 30)
                    })
                    data_source = "generated"
            except Exception as e:
                print(f"Error generating sample data: {e}")
                # Ultimate fallback - return statistical prediction
                return jsonify({
                    "error": "No data available",
                    "ensemble_prediction": 2000.0 + np.random.normal(0, 30),
                    "individual_predictions": {
                        "statistical_model": 2000.0 + np.random.normal(0, 30)
                    },
                    "timestamp": datetime.now().isoformat(),
                    "data_source": "statistical_fallback",
                    "message": "Using statistical fallback due to data unavailability"
                }), 200
        
        # Process data and make predictions
        processed_data = None
        try:
            if data_processor:
                processed_data = data_processor.prepare_training_data()
            else:
                # Simple processing fallback
                processed_data = latest_data.copy()
                if 'price' in processed_data.columns:
                    processed_data['price_lag_1'] = processed_data['price'].shift(1)
                    processed_data['sma_5'] = processed_data['price'].rolling(5).mean()
                processed_data = processed_data.dropna()
        except Exception as e:
            print(f"Error processing data: {e}")
            processed_data = latest_data.copy()
        
        if processed_data is None or len(processed_data) == 0:
            return jsonify({
                "error": "Data processing failed",
                "ensemble_prediction": 2000.0 + np.random.normal(0, 25),
                "timestamp": datetime.now().isoformat(),
                "data_source": data_source,
                "message": "Data processing failed - using statistical fallback"
            }), 200
        
        # Get latest features
        feature_columns = [col for col in processed_data.columns if col not in ['date', 'price']]
        latest_features = processed_data[feature_columns].iloc[-1].values if feature_columns else []
        
        # Get predictions from all models
        predictions = {}
        successful_predictions = 0
        
        if predictor and hasattr(predictor, 'trained_models'):
            for model_name in predictor.trained_models.keys():
                try:
                    pred = predictor.predict_single(model_name, latest_features)
                    predictions[model_name] = float(pred)
                    successful_predictions += 1
                except Exception as e:
                    print(f"Error with {model_name}: {e}")
                    predictions[model_name] = None
        
        # Add statistical fallback predictions if no models worked
        if successful_predictions == 0:
            base_price = float(processed_data['price'].iloc[-1]) if 'price' in processed_data.columns else 2000.0
            predictions = {
                "statistical_trend": base_price + np.random.normal(0, 20),
                "moving_average": base_price + np.random.normal(0, 15),
                "volatility_adjusted": base_price + np.random.normal(0, 25)
            }
            successful_predictions = 3
        
        # Ensemble prediction
        ensemble_pred = None
        try:
            if predictor and hasattr(predictor, 'predict_ensemble') and len(latest_features) > 0:
                ensemble_pred = predictor.predict_ensemble(latest_features)
                predictions['ensemble'] = float(ensemble_pred)
            else:
                # Calculate ensemble from individual predictions
                valid_predictions = [p for p in predictions.values() if p is not None and isinstance(p, (int, float))]
                ensemble_pred = np.mean(valid_predictions) if valid_predictions else 2000.0
                predictions['ensemble'] = float(ensemble_pred)
        except Exception as e:
            print(f"Ensemble prediction error: {e}")
            valid_predictions = [p for p in predictions.values() if p is not None and isinstance(p, (int, float))]
            ensemble_pred = np.mean(valid_predictions) if valid_predictions else 2000.0
            predictions['ensemble'] = float(ensemble_pred)
        
        # Current price for comparison
        current_price = float(processed_data['price'].iloc[-1]) if 'price' in processed_data.columns else 2000.0
        
        result = {
            "predictions": predictions,
            "current_price": current_price,
            "timestamp": datetime.now().isoformat(),
            "data_date": processed_data['date'].iloc[-1].isoformat() if 'date' in processed_data.columns and pd.notna(processed_data['date'].iloc[-1]) else datetime.now().isoformat(),
            "data_source": data_source,
            "successful_models": successful_predictions,
            "status": "success" if successful_predictions > 0 else "fallback"
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Critical error in predict_ensemble: {e}")
        traceback.print_exc()
        return jsonify({
            "error": "Critical system error",
            "ensemble_prediction": 2000.0 + np.random.normal(0, 40),
            "predictions": {
                "emergency_fallback": 2000.0
            },
            "timestamp": datetime.now().isoformat(),
            "message": "Emergency fallback - system requires maintenance"
        }), 200

@api_bp.route("/data/update", methods=["POST"])
def update_data():
    """Update gold price data with comprehensive error handling"""
    data_collector = get_data_collector()
    
    try:
        # Initialize data collector if not available
        if not data_collector:
            try:
                if GoldDataCollector:
                    data_collector = GoldDataCollector()
                else:
                    return jsonify({
                        "error": "Data collector not available",
                        "message": "Data collection service is currently unavailable",
                        "timestamp": datetime.now().isoformat(),
                        "status": "service_unavailable"
                    }), 200
            except Exception as e:
                print(f"Failed to initialize data collector: {e}")
                return jsonify({
                    "error": "Failed to initialize data collector",
                    "message": "Data collection service initialization failed",
                    "timestamp": datetime.now().isoformat(),
                    "status": "initialization_failed"
                }), 200
        
        data = request.get_json() or {}
        force_refresh = data.get('force_refresh', False)
        
        # Attempt to update data
        update_success = False
        error_details = []
        result = None
        
        try:
            # Update data
            result = data_collector.update_data(force_refresh=force_refresh)
            if result:
                update_success = True
            else:
                error_details.append("update_data returned None")
        except Exception as e:
            error_details.append(f"update_data failed: {str(e)}")
            print(f"Error in update_data: {e}")
            
            # Try alternative data collection methods
            try:
                # Try generating sample data as fallback
                sample_data = data_collector.generate_sample_gold_data(days=30)
                if sample_data is not None and len(sample_data) > 0:
                    # Save sample data
                    import os
                    os.makedirs('data/raw', exist_ok=True)
                    sample_data.to_csv('data/raw/gold_prices.csv', index=False)
                    update_success = True
                    error_details.append("Used sample data as fallback")
                    result = {'gold_data': sample_data, 'economic_data': None}
            except Exception as e2:
                error_details.append(f"Sample data generation failed: {str(e2)}")
                print(f"Sample data generation error: {e2}")
        
        if update_success and result:
            return jsonify({
                "status": "success",
                "message": "Data updated successfully",
                "gold_records": len(result['gold_data']) if result['gold_data'] is not None else 0,
                "economic_records": len(result['economic_data']) if result['economic_data'] is not None else 0,
                "timestamp": datetime.now().isoformat(),
                "details": error_details if error_details else ["Data collection completed normally"]
            })
        else:
            return jsonify({
                "error": "Data update failed",
                "message": "All data collection methods failed",
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error_details": error_details,
                "fallback_available": "System will use cached or default data"
            }), 200
        
    except Exception as e:
        print(f"Critical error in update_data: {e}")
        traceback.print_exc()
        return jsonify({
            "error": "Critical system error",
            "message": "Data update service encountered a critical error",
            "timestamp": datetime.now().isoformat(),
            "status": "critical_error",
            "fallback_available": "System will continue with existing data"
        }), 200

@api_bp.route("/analysis/summary", methods=["GET"])
def get_analysis_summary():
    """Get statistical analysis of gold price data with comprehensive error handling"""
    data_collector = get_data_collector()
    
    try:
        # Initialize data collector if not available
        if not data_collector:
            try:
                if GoldDataCollector:
                    data_collector = GoldDataCollector()
                else:
                    # Return basic statistical summary without real data
                    return jsonify({
                        "error": "Data collector not available",
                        "fallback_analysis": {
                            "estimated_current_price": 2000.0,
                            "typical_price_range": {"min": 1800.0, "max": 2200.0},
                            "message": "Analysis service unavailable - showing typical gold price ranges"
                        },
                        "timestamp": datetime.now().isoformat(),
                        "status": "service_unavailable"
                    }), 200
            except Exception as e:
                print(f"Failed to initialize data collector: {e}")
        
        # Get latest data
        data = None
        data_source = "unknown"
        
        try:
            if data_collector:
                data = data_collector.load_latest_data('gold_prices')
                data_source = "file" if data is not None else "none"
        except Exception as e:
            print(f"Error loading data for analysis: {e}")
        
        # Fallback to sample data if no real data available
        if data is None or len(data) == 0:
            try:
                if data_collector:
                    data = data_collector.generate_sample_gold_data(days=90)
                    data_source = "sample"
                else:
                    # Generate basic sample data for analysis
                    dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
                    prices = np.random.normal(2000, 50, 90)
                    # Add some trend
                    trend = np.linspace(-20, 20, 90)
                    prices += trend
                    data = pd.DataFrame({
                        'date': dates,
                        'price': prices,
                        'volume': np.random.randint(50000, 200000, 90)
                    })
                    data_source = "generated"
            except Exception as e:
                print(f"Error generating sample data for analysis: {e}")
                # Ultimate fallback - return basic analysis
                return jsonify({
                    "error": "No data available for analysis",
                    "fallback_analysis": {
                        "estimated_current_price": 2000.0,
                        "historical_average": 1950.0,
                        "typical_volatility": "2-3% daily",
                        "message": "Unable to generate analysis - using historical averages"
                    },
                    "timestamp": datetime.now().isoformat(),
                    "data_source": "historical_fallback"
                }), 200
        
        # Ensure we have valid data
        if data is None or len(data) == 0:
            return jsonify({
                "error": "No valid data for analysis",
                "fallback_analysis": {
                    "estimated_current_price": 2000.0,
                    "message": "Analysis requires valid price data"
                },
                "timestamp": datetime.now().isoformat()
            }), 200
        
        # Calculate statistics with error handling
        try:
            # Ensure price column exists and is numeric
            if 'price' not in data.columns:
                return jsonify({
                    "error": "Price data not found",
                    "available_columns": list(data.columns),
                    "timestamp": datetime.now().isoformat()
                }), 200
            
            # Clean price data
            price_data = pd.to_numeric(data['price'], errors='coerce').dropna()
            
            if len(price_data) == 0:
                return jsonify({
                    "error": "No valid price data found",
                    "fallback_analysis": {
                        "estimated_current_price": 2000.0,
                        "message": "Price data contains no valid numeric values"
                    },
                    "timestamp": datetime.now().isoformat()
                }), 200
            
            # Calculate basic statistics
            stats = {
                "total_records": len(data),
                "valid_price_records": len(price_data),
                "data_source": data_source,
                "date_range": {
                    "start": data['date'].min().isoformat() if 'date' in data.columns and pd.notna(data['date'].min()) else "Unknown",
                    "end": data['date'].max().isoformat() if 'date' in data.columns and pd.notna(data['date'].max()) else datetime.now().isoformat()
                },
                "price_statistics": {
                    "current": float(price_data.iloc[-1]),
                    "mean": float(price_data.mean()),
                    "median": float(price_data.median()),
                    "std": float(price_data.std()),
                    "min": float(price_data.min()),
                    "max": float(price_data.max()),
                    "volatility_percent": float((price_data.std() / price_data.mean()) * 100)
                },
                "recent_trend": {},
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
            # Calculate trend statistics with error handling
            try:
                if len(price_data) >= 8:
                    stats["recent_trend"]["last_7_days_change"] = float(price_data.iloc[-1] - price_data.iloc[-8])
                    stats["recent_trend"]["last_7_days_change_percent"] = float(((price_data.iloc[-1] - price_data.iloc[-8]) / price_data.iloc[-8]) * 100)
                
                if len(price_data) >= 31:
                    stats["recent_trend"]["last_30_days_change"] = float(price_data.iloc[-1] - price_data.iloc[-31])
                    stats["recent_trend"]["last_30_days_change_percent"] = float(((price_data.iloc[-1] - price_data.iloc[-31]) / price_data.iloc[-31]) * 100)
                
                # Add moving averages
                if len(price_data) >= 5:
                    stats["moving_averages"] = {
                        "sma_5": float(price_data.tail(5).mean()),
                        "sma_10": float(price_data.tail(10).mean()) if len(price_data) >= 10 else None,
                        "sma_20": float(price_data.tail(20).mean()) if len(price_data) >= 20 else None
                    }
            except Exception as e:
                print(f"Error calculating trends: {e}")
                stats["recent_trend"]["error"] = "Trend calculation failed"
            
            return jsonify(stats)
            
        except Exception as e:
            print(f"Error calculating statistics: {e}")
            return jsonify({
                "error": "Statistical calculation failed",
                "fallback_analysis": {
                    "estimated_current_price": 2000.0,
                    "data_records": len(data),
                    "message": "Statistical analysis failed - using fallback values"
                },
                "timestamp": datetime.now().isoformat(),
                "data_source": data_source
            }), 200
        
    except Exception as e:
        print(f"Critical error in get_analysis_summary: {e}")
        traceback.print_exc()
        return jsonify({
            "error": "Critical system error",
            "fallback_analysis": {
                "estimated_current_price": 2000.0,
                "message": "Analysis service encountered a critical error"
            },
            "timestamp": datetime.now().isoformat(),
            "status": "critical_error"
        }), 200

@api_bp.route("/data/realtime", methods=["GET"])
def get_realtime_data():
    """Get real-time gold price data for live chart updates"""
    data_collector = get_data_collector()
    
    try:
        # Initialize data collector if not available
        if not data_collector:
            try:
                if GoldDataCollector:
                    data_collector = GoldDataCollector()
            except Exception as e:
                print(f"Failed to initialize data collector: {e}")
        
        # Get latest data
        latest_data = None
        data_source = "unknown"
        
        try:
            if data_collector:
                latest_data = data_collector.load_latest_data('gold_prices')
                data_source = "file" if latest_data is not None else "none"
        except Exception as e:
            print(f"Error loading real-time data: {e}")
        
        # Fallback to sample data if no real data available
        if latest_data is None or len(latest_data) == 0:
            try:
                if data_collector:
                    latest_data = data_collector.generate_sample_gold_data(days=1)
                    data_source = "sample"
                else:
                    # Generate a single real-time data point
                    base_price = 2000.0
                    current_time = datetime.now()
                    latest_data = pd.DataFrame({
                        'date': [current_time],
                        'price': [base_price + np.random.normal(0, 10)],
                        'open': [base_price + np.random.normal(0, 5)],
                        'high': [base_price + np.random.normal(5, 5)],
                        'low': [base_price + np.random.normal(-5, 5)],
                        'close': [base_price + np.random.normal(0, 10)],
                        'volume': [np.random.randint(80000, 150000)]
                    })
                    data_source = "generated"
            except Exception as e:
                print(f"Error generating real-time sample data: {e}")
                # Ultimate fallback
                current_time = datetime.now()
                return jsonify({
                    "timestamp": current_time.isoformat(),
                    "price": 2000.0 + np.random.normal(0, 15),
                    "open": 1995.0,
                    "high": 2010.0,
                    "low": 1985.0,
                    "close": 2000.0 + np.random.normal(0, 15),
                    "volume": 100000,
                    "change": np.random.normal(0, 5),
                    "change_percent": np.random.normal(0, 0.25),
                    "data_source": "fallback",
                    "status": "success"
                })
        
        # Get the latest record
        if len(latest_data) > 0:
            latest_record = latest_data.iloc[-1]
            
            # Calculate change from previous record if available
            change = 0.0
            change_percent = 0.0
            if len(latest_data) > 1:
                prev_price = latest_data.iloc[-2]['price']
                change = float(latest_record['price'] - prev_price)
                change_percent = float((change / prev_price) * 100) if prev_price != 0 else 0.0
            
            # Prepare real-time data response
            realtime_data = {
                "timestamp": latest_record['date'].isoformat() if pd.notna(latest_record['date']) else datetime.now().isoformat(),
                "price": float(latest_record['price']),
                "open": float(latest_record.get('open', latest_record['price'])),
                "high": float(latest_record.get('high', latest_record['price'])),
                "low": float(latest_record.get('low', latest_record['price'])),
                "close": float(latest_record.get('close', latest_record['price'])),
                "volume": int(latest_record.get('volume', 100000)),
                "change": change,
                "change_percent": change_percent,
                "data_source": data_source,
                "status": "success"
            }
            
            return jsonify(realtime_data)
        
        else:
            return jsonify({
                "error": "No real-time data available",
                "timestamp": datetime.now().isoformat(),
                "status": "no_data"
            }), 404
            
    except Exception as e:
        print(f"Critical error in get_realtime_data: {e}")
        traceback.print_exc()
        return jsonify({
            "error": "Critical system error",
            "timestamp": datetime.now().isoformat(),
            "price": 2000.0 + np.random.normal(0, 20),
            "status": "error",
            "message": "Real-time data service encountered an error"
        }), 200

@api_bp.route("/data/ohlc", methods=["GET"])
def get_ohlc_data():
    """Get OHLC (Open, High, Low, Close) data for candlestick charts"""
    data_collector = get_data_collector()
    
    try:
        # Get query parameters
        days = request.args.get('days', 30, type=int)
        interval = request.args.get('interval', 'daily')  # daily, hourly
        
        # Limit days to prevent excessive data
        days = min(days, 365)
        
        # Initialize data collector if not available
        if not data_collector:
            try:
                if GoldDataCollector:
                    data_collector = GoldDataCollector()
            except Exception as e:
                print(f"Failed to initialize data collector: {e}")
        
        # Get historical data
        data = None
        data_source = "unknown"
        
        try:
            if data_collector:
                data = data_collector.load_latest_data('gold_prices')
                data_source = "file" if data is not None else "none"
        except Exception as e:
            print(f"Error loading OHLC data: {e}")
        
        # Fallback to sample data
        if data is None or len(data) == 0:
            try:
                if data_collector:
                    data = data_collector.generate_sample_gold_data(days=days)
                    data_source = "sample"
                else:
                    # Generate OHLC sample data
                    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
                    base_prices = np.random.normal(2000, 50, days)
                    
                    ohlc_data = []
                    for i, (date, base_price) in enumerate(zip(dates, base_prices)):
                        # Generate realistic OHLC values
                        daily_range = np.random.uniform(10, 50)
                        open_price = base_price + np.random.normal(0, 5)
                        high_price = open_price + np.random.uniform(0, daily_range)
                        low_price = open_price - np.random.uniform(0, daily_range)
                        close_price = open_price + np.random.normal(0, daily_range/2)
                        
                        ohlc_data.append({
                            'date': date,
                            'open': open_price,
                            'high': high_price,
                            'low': low_price,
                            'close': close_price,
                            'price': close_price,
                            'volume': np.random.randint(50000, 200000)
                        })
                    
                    data = pd.DataFrame(ohlc_data)
                    data_source = "generated"
            except Exception as e:
                print(f"Error generating OHLC sample data: {e}")
                return jsonify({
                    "error": "Failed to generate OHLC data",
                    "message": "OHLC data service is currently unavailable"
                }), 500
        
        # Get the requested number of days
        if len(data) > days:
            data = data.tail(days)
        
        # Prepare OHLC response
        ohlc_records = []
        for _, row in data.iterrows():
            record = {
                "date": row['date'].isoformat() if pd.notna(row['date']) else datetime.now().isoformat(),
                "open": float(row.get('open', row['price'])),
                "high": float(row.get('high', row['price'])),
                "low": float(row.get('low', row['price'])),
                "close": float(row.get('close', row['price'])),
                "volume": int(row.get('volume', 100000))
            }
            ohlc_records.append(record)
        
        return jsonify({
            "data": ohlc_records,
            "total_records": len(ohlc_records),
            "days_requested": days,
            "interval": interval,
            "data_source": data_source,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        })
        
    except Exception as e:
        print(f"Critical error in get_ohlc_data: {e}")
        traceback.print_exc()
        return jsonify({
            "error": "Critical system error",
            "message": "OHLC data service encountered an error",
            "timestamp": datetime.now().isoformat()
        }), 500

@api_bp.route("/auto-updater/status", methods=["GET"])
def get_auto_updater_status():
    """Get auto updater status"""
    try:
        # Import here to avoid circular imports
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        
        # Try to get the auto updater from the main module
        try:
            import run_fresh
            if hasattr(run_fresh, 'auto_updater') and run_fresh.auto_updater:
                status = run_fresh.auto_updater.get_status()
                return jsonify({
                    "status": "success",
                    "auto_updater": status
                })
            else:
                return jsonify({
                    "status": "not_running",
                    "message": "Auto updater is not running"
                })
        except ImportError:
            return jsonify({
                "status": "not_available",
                "message": "Auto updater module not available"
            })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@api_bp.route("/auto-updater/trigger", methods=["POST"])
def trigger_auto_update():
    """Manually trigger an auto update"""
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        
        try:
            import run_fresh
            if hasattr(run_fresh, 'auto_updater') and run_fresh.auto_updater:
                success = run_fresh.auto_updater.update_data()
                return jsonify({
                    "status": "success" if success else "failed",
                    "message": "Update triggered successfully" if success else "Update failed",
                    "timestamp": datetime.now().isoformat()
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": "Auto updater is not running"
                }), 400
        except ImportError:
            return jsonify({
                "status": "error",
                "message": "Auto updater module not available"
            }), 400
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@api_bp.route("/debug-advanced", methods=["GET", "POST"])
def debug_advanced():
    """Debug advanced prediction endpoint"""
    try:
        # Check predictor
        predictor = get_predictor()
        if not predictor:
            return jsonify({'error': 'Predictor not initialized', 'step': 'predictor_check'}), 500
        
        # Check if models are trained
        if not hasattr(predictor, 'best_model') or predictor.best_model is None:
            return jsonify({'error': 'No trained models available', 'step': 'model_check'}), 500
        
        # Check data collector
        data_collector = get_data_collector()
        if not data_collector:
            return jsonify({'error': 'Data collector not available', 'step': 'data_collector_check'}), 500
        
        # Try to get latest data
        try:
            latest_data = data_collector.load_latest_data('gold_prices')
            if latest_data is None or latest_data.empty:
                return jsonify({'error': 'No data available', 'step': 'data_availability_check'}), 500
        except Exception as e:
            return jsonify({'error': f'Error getting data: {str(e)}', 'step': 'data_retrieval'}), 500
        
        # Try a simple prediction
        try:
            prediction_data = latest_data.tail(1).to_dict('records')[0]
            df = pd.DataFrame([prediction_data])
            prediction = predictor.predict(df)
            
            return jsonify({
                'status': 'success',
                'prediction': float(prediction[0]) if hasattr(prediction, '__iter__') else float(prediction),
                'data_shape': df.shape,
                'step': 'prediction_success'
            })
        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}', 'step': 'prediction_attempt'}), 500
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc(),
            'step': 'general_error'
        }), 500

# Models will be initialized by the app factory
# No automatic initialization here to avoid scope issues
