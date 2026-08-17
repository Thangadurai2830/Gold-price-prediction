from flask import Flask, render_template

def create_app():
    app = Flask(__name__)
    app.config['DEBUG'] = True

    # Import blueprints
    from app.api import api_bp
    from app.api.live_chat import live_chat_bp
    
    app.register_blueprint(api_bp, url_prefix="/api")
    app.register_blueprint(live_chat_bp)
    
    # Initialize models after blueprint registration
    with app.app_context():
        from app.api import initialize_models
        try:
            success = initialize_models(app)
            if success:
                print("✓ API models initialized successfully in app context")
                # Debug: Check if models are actually stored
                if hasattr(app, 'predictor'):
                    print(f"✓ Predictor stored in app: {app.predictor}")
                    if hasattr(app.predictor, 'trained_models'):
                        print(f"✓ Models in app predictor: {list(app.predictor.trained_models.keys())}")
                else:
                    print("⚠ Predictor not found in app")
            else:
                print("⚠ API models initialization failed - using fallback mode")
        except Exception as e:
            print(f"Warning: Could not initialize models in app context: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")

    @app.route("/")
    def home():
        return render_template("index.html")
    
    @app.route("/dashboard")
    def dashboard():
        return render_template("index.html")

    return app
