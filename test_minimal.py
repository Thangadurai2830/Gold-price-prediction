#!/usr/bin/env python3
"""
Minimal Flask app to test routing
"""

from flask import Flask, jsonify

def create_minimal_app():
    app = Flask(__name__)
    app.config['DEBUG'] = True

    @app.route("/")
    def home():
        return jsonify({"message": "Home works"})
    
    @app.route("/test")
    def test():
        return jsonify({"status": "ok", "message": "Test endpoint works"})
    
    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok", "message": "Health endpoint works"})

    return app

if __name__ == "__main__":
    app = create_minimal_app()
    print("Starting minimal Flask app...")
    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)