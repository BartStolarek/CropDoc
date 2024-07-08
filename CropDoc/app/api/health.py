from flask import Blueprint, jsonify

health_blueprint = Blueprint('health', __name__)

@health_blueprint.route('/health', methods=['GET'])
def get_model():
    # Return 200 OK with healthy status
    return jsonify({'status': 'healthy'})