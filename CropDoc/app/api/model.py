from flask import Blueprint, jsonify

model_blueprint = Blueprint('model', __name__)

@model_blueprint.route('/model', methods=['GET'])
def get_model():
    return jsonify({'model': 'model data'})