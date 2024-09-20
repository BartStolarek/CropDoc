import os

import numpy as np
from flask import Blueprint, jsonify, request
from flask_restx import Namespace, Resource, fields
from loguru import logger
from werkzeug.utils import secure_filename

from app import main_api
from app.config import AppConfig
from app.handler.pipeline import handle_predict

pipeline_blueprint = Blueprint('pipeline', __name__)
pipeline_ns = Namespace('pipeline', description='Pipeline operations for disease classification')

main_api.add_namespace(pipeline_ns)

# Define the pipeline data structure
pipeline_data = pipeline_ns.model(
    'PipelineData', {
        'pipeline': fields.String(description='Information about the current disease classification pipeline')
    }
)

@pipeline_ns.route('/')
class PipelineResource(Resource):
    @pipeline_ns.doc(
        'get_pipeline',
        description='Retrieve information about the current disease classification pipeline'
    )
    @pipeline_ns.marshal_with(pipeline_data)
    def get(self):
        """
        Get pipeline data
        
        Returns:
            dict: A dictionary containing information about the current disease classification pipeline
        """
        return {'pipeline': 'Current pipeline configuration and status'}

@pipeline_ns.route('/predict')
class PredictResource(Resource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize pipeline if necessary
        # self.pipeline = load_your_pipeline()

    @pipeline_ns.doc(
        'predict',
        description='Predict the class of the uploaded image using the disease classification pipeline'
    )
    @pipeline_ns.expect(pipeline_ns.parser().add_argument('file', location='files', type='file', required=True))
    @pipeline_ns.response(200, 'Success', fields.Raw(description='Prediction results'))
    @pipeline_ns.response(400, 'Bad Request')
    @pipeline_ns.response(500, 'Internal Server Error')
    def post(self):
        """
        Predict the class of the uploaded image
        
        This endpoint accepts an image file upload and uses the disease classification pipeline to predict its class.
        
        Returns:
            dict: A dictionary containing the prediction results
        
        Raises:
            400 Bad Request: If no file is uploaded or the file is invalid
            500 Internal Server Error: If file upload or processing fails
        """
        logger.info('Received POST request to predict')

        if 'file' not in request.files:
            logger.debug('No file key in request.files')
            return {'message': 'No file key'}, 400

        file = request.files['file']
        if file.filename == '':
            logger.debug('No selected file')
            return {'message': 'No selected file'}, 400

        if file:
            filename = secure_filename(file.filename)
            temp_dir = os.path.join(AppConfig.DATA_DIR, 'tmp')
            os.makedirs(temp_dir, exist_ok=True)

            file_path = os.path.join(temp_dir, filename)
            file.save(file_path)

            logger.info(f'File saved to {file_path}')

            config = 'resnet50-v3_1'
            kwargs = {'image_path': file_path}

            results = handle_predict(pipeline_config=config, **kwargs)

            def convert_to_serializable(obj):
                if isinstance(obj, np.generic):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {key: convert_to_serializable(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                else:
                    return obj
            
            serializable_results = convert_to_serializable(results)

            os.remove(file_path)

            logger.info(f'Responding with prediction: {results}...')

            return jsonify(serializable_results)

        return {'message': 'File upload failed'}, 500