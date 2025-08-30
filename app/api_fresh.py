#!/usr/bin/env python3
"""
Fresh API blueprint to bypass import caching issues
"""

from flask import Blueprint, jsonify, current_app
from datetime import datetime
import os
import logging

# Create fresh blueprint
api_fresh_bp = Blueprint('api_fresh', __name__)

def get_predictor():
    """Get the predictor from current app context"""
    return getattr(current_app, 'predictor', None)

def initialize_models():
    """Initialize ML models and store in app context"""
    try:
        from ml.models import GoldPricePredictor
        
        # Initialize predictor
        predictor = GoldPricePredictor()
        
        # Load existing models
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
        success = predictor.load_models(models_dir)
        
        if success:
            # Store in app context
            current_app.predictor = predictor
            print(f"✓ Fresh API - GoldPricePredictor object and its models successfully stored in app context")
            
            if hasattr(predictor, 'trained_models'):
                model_count = len(predictor.trained_models)
                model_names = list(predictor.trained_models.keys())
                print(f"✓ Fresh API - {model_count} models loaded: {model_names}")
            
            return True
        else:
            print("⚠ Fresh API - Failed to load models")
            return False
            
    except Exception as e:
        print(f"Error in fresh API initialize_models: {e}")
        import traceback
        traceback.print_exc()
        return False

@api_fresh_bp.route("/ping", methods=["GET"])
def ping():
    """Health check endpoint"""
    return jsonify({
        "message": "pong", 
        "status": "ok",
        "fresh_api": True,
        "timestamp": datetime.now().isoformat()
    })

@api_fresh_bp.route("/health", methods=["GET"])
def health_check():
    """Comprehensive health check with model status"""
    predictor = get_predictor()
    
    # Determine model status
    models_count = 0
    if predictor and hasattr(predictor, 'trained_models'):
        models_count = len(predictor.trained_models)
    
    # Determine overall system health
    if models_count == 0:
        overall_status = "degraded"  # System running but no models
    elif models_count < 3:
        overall_status = "partial"   # Some models loaded
    else:
        overall_status = "healthy"   # Good model coverage
    
    # Build response
    status = {
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "models_loaded": models_count > 0,  # True if any models are loaded
        "models_count": models_count,
        "version": "1.0.0",
        "fresh_api": True,
        "debug": {
            "predictor_exists": predictor is not None,
            "predictor_type": str(type(predictor)) if predictor else None,
            "has_trained_models": hasattr(predictor, 'trained_models') if predictor else False,
            "model_keys": list(predictor.trained_models.keys()) if predictor and hasattr(predictor, 'trained_models') else [],
        },
        "system_info": {
            "predictor_initialized": predictor is not None,
            "fallback_available": True
        }
    }
    
    if predictor and hasattr(predictor, 'trained_models'):
        available_models = list(predictor.trained_models.keys())
        status["available_models"] = available_models
        status["prophet_model"] = hasattr(predictor, 'prophet_model') and predictor.prophet_model is not None
        status["feature_columns_count"] = len(predictor.feature_columns) if hasattr(predictor, 'feature_columns') else 0
        
        # Check if using fallback system
        if "fallback_linear" in available_models:
            status["using_fallback"] = True
            status["warning"] = "System is using fallback prediction model"
        else:
            status["using_fallback"] = False
    else:
        status["error"] = "Predictor not initialized - using statistical fallbacks"
        status["available_models"] = []
        status["prophet_model"] = False
        status["using_fallback"] = True
    
    return jsonify(status)

@api_fresh_bp.route("/debug", methods=["GET"])
def debug_info():
    """Debug endpoint to check system state"""
    predictor = get_predictor()
    
    debug_data = {
        "timestamp": datetime.now().isoformat(),
        "fresh_api": True,
        "predictor": {
            "exists": predictor is not None,
            "type": str(type(predictor)) if predictor else None,
            "has_trained_models": hasattr(predictor, 'trained_models') if predictor else False,
        },
        "app_context": {
            "has_predictor_attr": hasattr(current_app, 'predictor'),
            "predictor_same_object": getattr(current_app, 'predictor', None) is predictor,
        }
    }
    
    if predictor and hasattr(predictor, 'trained_models'):
        debug_data["models"] = {
            "count": len(predictor.trained_models),
            "names": list(predictor.trained_models.keys()),
            "types": {name: str(type(model)) for name, model in predictor.trained_models.items()}
        }
        
        if hasattr(predictor, 'feature_columns'):
            debug_data["features"] = {
                "count": len(predictor.feature_columns),
                "columns": predictor.feature_columns[:10] if len(predictor.feature_columns) > 10 else predictor.feature_columns
            }
    
    return jsonify(debug_data)