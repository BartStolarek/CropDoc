# app/api/model.py

from flask import Blueprint, request, jsonify
from flask_restx import Namespace, Resource, fields
from werkzeug.utils import secure_filename
import os

from app import main_api
from app.handler.pipeline import handle_pipeline
from loguru import logger
from app.config import AppConfig

model_blueprint = Blueprint('model', __name__)
model_ns = Namespace('model', description='Model operations')

main_api.add_namespace(model_ns)

# Define the model data structure (this can be updated according to your needs)
model_data = model_ns.model('ModelData', {
    'model': fields.String(description='Model data')
})

@model_ns.route('/')
class ModelResource(Resource):
    @model_ns.doc(
        'get_model',
        description='Retrieve information about the current disease classification model'
    )
    @model_ns.marshal_with(model_data)
    def get(self):
        """Get model data"""
        return {'model': 'model_data'}

@model_ns.route('/predict')
class PredictResource(Resource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # You might need to initialize your model here, if it's not already initialized
        # self.model = load_your_model()

    @model_ns.doc(
        'predict',
        description='Predict the class of the uploaded image'
    )
    def post(self):
        """Predict the class of the uploaded image"""
        
        print('PredictResource post() method called')

        if 'file' not in request.files:
            logger.debug('No file key in request.files')
            return {'message': 'No file key'}, 400
        
        file = request.files['file']
        if file.filename == '':
            logger.debug('No selected file')
            return {'message': 'No selected file'}, 400
        
        if file:
            # Secure the filename and ensure the directory exists
            filename = secure_filename(file.filename)
            temp_dir = os.path.join(AppConfig.DATA_DIR, 'tmp')
            os.makedirs(temp_dir, exist_ok=True)  # Create the directory if it does not exist
            
            file_path = os.path.join(temp_dir, filename)
            file.save(file_path)
            
            logger.info(f'File saved to {file_path}')
            
            # Load the image and make prediction (pseudo-code)
            # image = load_image(file_path)
            # prediction = self.model.predict(image)
            config = 'resnet50-split'
            pipeline_file = 'resnet50-split'
            method = 'predict_one'
            kwargs = {
                'file_path': file_path
            }
            
            results = handle_pipeline(file=pipeline_file, method=method, model_config=config, **kwargs)
            
            
            # # For demo purposes, we'll just return a dummy response
            # prediction = {'class': 'dummy_class', 'confidence': 0.95}

            # Remove the file after processing
            os.remove(file_path)
            
            logger.info(f'Responding with prediction: {results}')
            
            return jsonify(results)

        return {'message': 'File upload failed'}, 500
