from flask import Blueprint
from flask_restx import Namespace, Resource, fields

from app import main_api

health_blueprint = Blueprint('health', __name__)
health_ns = Namespace('health', description='Server Health')

main_api.add_namespace(health_ns)

health_data = health_ns.model(
    'HealthData',
    {'status': fields.String(description='Server health status')})


@health_ns.route('/')
class HealthResource(Resource):

    @health_ns.doc('get_health')
    @health_ns.marshal_with(health_data)
    def get(self):
        """Get server health status"""
        return {'status': 'healthy'}
