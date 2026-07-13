"""
app.py
Flask application factory for the BI Dashboard prototype.
Run with:  python app.py
"""

import sys
from pathlib import Path
# This adds the root folder (E:\DEPI Project) to Python's search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os

from flask import Flask, render_template, request

from blueprints.dashboard import dashboard_bp
from blueprints.insights import insights_bp
from blueprints.forecasts import forecasts_bp
from blueprints.data_upload import data_upload_bp
from blueprints.alerts import alerts_bp
from blueprints.reports import reports_bp
from blueprints.settings import settings_bp


def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-key-change-me")
    app.config["ENV"] = os.getenv("FLASK_ENV", "production")
    app.config["DEBUG"] = os.getenv("FLASK_DEBUG", "0") == "1"

    # --- ADDED: Global Request Logger ---
    @app.before_request
    def log_request_info():
        print(f"DEBUG: Incoming request to: {request.url}")
    # ------------------------------------

    # Register blueprints (modular routing)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(insights_bp)
    app.register_blueprint(forecasts_bp)
    app.register_blueprint(data_upload_bp)
    app.register_blueprint(alerts_bp)
    app.register_blueprint(reports_bp)
    app.register_blueprint(settings_bp)

    @app.errorhandler(404)
    def not_found(e):
        return render_template("404.html"), 404

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=8080)