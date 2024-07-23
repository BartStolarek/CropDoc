# app/api/model.py

from flask import Blueprint
from flask_restx import Namespace, Resource, fields
from app import main_api

model_blueprint = Blueprint('model', __name__)
model_ns = Namespace('model', description='Model operations')

main_api.add_namespace(model_ns)

model_data = model_ns.model('ModelData', {
    'model': fields.String(description='Model data')
})


@model_ns.route('/')
class ModelResource(Resource):
    @model_ns.doc('get_model', description='Retrieve information about the current disease classification model')
    @model_ns.marshal_with(model_data)
    def get(self):
        """Get model data"""
        return {'model': 'model_data'}
