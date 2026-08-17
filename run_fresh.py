#!/usr/bin/env python3
"""
Fresh Flask app runner to bypass caching issues
"""

import os
import sys
from flask import Flask, jsonify
import atexit

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import auto updater
from auto_updater import AutoUpdater

# Global auto updater instance
auto_updater = None

def get_auto_updater():
    """Get the global auto updater instance"""
    return auto_updater

def create_fresh_app():
    import os
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app', 'templates')
    app = Flask(__name__, template_folder=template_dir)
    app.config['DEBUG'] = True
    print(f"✓ Flask app created: {app}")
    print(f"✓ Template directory: {template_dir}")

    # Import and register original blueprint
    try:
        from app.api import api_bp, initialize_models
        print(f"✓ Blueprint imported: {api_bp}")
        app.register_blueprint(api_bp, url_prefix="/api")
        print(f"✓ Blueprint registered with prefix /api")
    except Exception as e:
        print(f"✗ Error importing/registering blueprint: {e}")
        import traceback
        traceback.print_exc()
    
    # Initialize models
    with app.app_context():
        from app.api import initialize_models
        try:
            success = initialize_models()
            if success:
                print("✓ Fresh app - models initialized successfully")
                if hasattr(app, 'predictor'):
                    print(f"✓ Fresh app - predictor: {app.predictor}")
                    if hasattr(app.predictor, 'trained_models'):
                        print(f"✓ Fresh app - models: {list(app.predictor.trained_models.keys())}")
            else:
                print("⚠ Fresh app - models initialization failed")
        except Exception as e:
            print(f"Error in fresh app: {e}")

    @app.route("/")
    def home():
        from flask import render_template
        return render_template("index.html")
    
    @app.route("/dashboard")
    def dashboard():
        from flask import render_template
        return render_template("index.html")
    
    @app.route("/test-direct")
    def test_direct():
        """Direct test endpoint on main app"""
        return jsonify({"status": "ok", "message": "Direct endpoint works"})
    
    print(f"✓ Routes registered: {[rule.rule for rule in app.url_map.iter_rules()]}")
    return app

def start_auto_updater():
    """Start the auto updater"""
    global auto_updater
    auto_updater = AutoUpdater(update_interval_minutes=15)
    auto_updater.start()
    print("✓ Auto updater started with 15-minute intervals")

def cleanup():
    """Cleanup function for auto updater"""
    global auto_updater
    if auto_updater:
        auto_updater.stop()
        print("✓ Auto updater stopped")

if __name__ == "__main__":
    app = create_fresh_app()
    
    # Start auto updater
    start_auto_updater()
    
    # Register cleanup function
    atexit.register(cleanup)
    
    try:
        app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\nShutting down...")
        cleanup()