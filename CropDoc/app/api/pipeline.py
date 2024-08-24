# app/api/pipeline.py

from flask import Blueprint, request, jsonify
from flask_restx import Namespace, Resource, fields
from werkzeug.utils import secure_filename
import os

from app import main_api
from app.handler.pipeline import handle_pipeline
from loguru import logger
from app.config import AppConfig

pipeline_blueprint = Blueprint('pipeline', __name__)
pipeline_ns = Namespace('pipeline', description='pipeline operations')

main_api.add_namespace(pipeline_ns)

# Define the pipeline data structure (this can be updated according to your needs)
pipeline_data = pipeline_ns.pipeline('pipelineData', {
    'pipeline': fields.String(description='pipeline data')
})


@pipeline_ns.route('/')
class pipelineResource(Resource):
    @pipeline_ns.doc(
        'get_pipeline',
        description='Retrieve information about the current disease classification pipeline'
    )
    @pipeline_ns.marshal_with(pipeline_data)
    def get(self):
        """Get pipeline data"""
        return {'pipeline': 'pipeline_data'}


@pipeline_ns.route('/predict')
class PredictResource(Resource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # You might need to initialize your pipeline here, if it's not already initialized
        # self.pipeline = load_your_pipeline()

    @pipeline_ns.doc(
        'predict',
        description='Predict the class of the uploaded image'
    )
    def post(self):
        """Predict the class of the uploaded image"""
        
        logger.info('Received POST request to predict')

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
            
            # Generate random 25 character string
            token = os.urandom(25).hex()
            
            # Add token to filename
            filename = f'{filename}_{token}'
            
            file_path = os.path.join(temp_dir, filename)
            file.save(file_path)
            
            logger.info(f'File saved to {file_path}')
            
            # Load the image and make prediction (pseudo-code)
            # image = load_image(file_path)
            # prediction = self.pipeline.predict(image)
            config = 'resnet50-split'
            pipeline_file = 'pipeline'
            method = 'predict'
            kwargs = {
                'image_path': file_path
            }
            
            results = handle_pipeline(file=pipeline_file, method=method, pipeline_config=config, **kwargs)
            
            # Remove the file after processing
            os.remove(file_path)
            
            logger.info(f'Responding with prediction: {results}')
            
            return jsonify(results)

        return {'message': 'File upload failed'}, 500
