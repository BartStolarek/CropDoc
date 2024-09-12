# app/__init__.py

import os
from pathlib import Path

from flask import Flask
from flask_cors import CORS
from flask_restx import Api

from app.config import AppConfig
from app.utility.logger import setup_logger

main_api = Api(version='1.0',
               title='CropDoc APIs',
               description='TODO: Add in description',
               doc='/doc/',
               default='CropDoc',
               default_label='CropDoc operations')


def create_app(config=AppConfig):
    app = Flask(__name__)
    app.config.from_object(config)

    allowed_origins = os.getenv('ALLOWED_ORIGINS',
                                'http://localhost:3000,http://frontend:3000').split(',')
    CORS(app,
         supports_credentials=True,
         origins=allowed_origins,
         resources={r"/*": {"origins": "*"}},
         methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],
         allow_headers=[
             'Content-Type', 'Authorization', 'ngrok-skip-browser-warning'
         ])

    # Create a pretty logger
    setup_logger()

    # Import and register blueprints
    from app.api.health import health_blueprint
    app.register_blueprint(health_blueprint)
    from app.api.pipeline import pipeline_blueprint
    app.register_blueprint(pipeline_blueprint)

    main_api.init_app(app)

    # Get current files path, and parent's path
    current_path = Path(__file__).parent
    parent_path = current_path.parent

    # Check if 'data' folder exists in parent path
    if not (parent_path / 'data').exists():
        print(
            "Data folder not found. Creating 'data' (not tracked on git) folder."
        )
        (parent_path / 'data').mkdir()
        
    # Create the home page to just be a blank page that says 'Backend Server is running'
    @app.route('/')
    def home():
        return 'Backend Server is running'

    return app


if __name__ == '__main__':
    app = create_app(AppConfig)
    app.run(host='0.0.0.0', port=5000, debug=True)
