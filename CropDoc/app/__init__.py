# app/__init__.py

from flask import Flask
from app.config import AppConfig
from loguru import logger
from app.utility.logger import setup_logger


def create_app(config=AppConfig):
    app = Flask(__name__)
    app.config.from_object(config)
    
    # Create a pretty logger
    setup_logger()

    # Import and register blueprints
    from app.api.health import health_blueprint
    app.register_blueprint(health_blueprint)
    from app.api.model import model_blueprint
    app.register_blueprint(model_blueprint)
    
    return app

if __name__ == '__main__':
    app = create_app(AppConfig)
    app.run(host='0.0.0.0', port=5000, debug=True)
