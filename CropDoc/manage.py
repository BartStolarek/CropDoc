import os
import click
from flask import Flask
from flask.cli import FlaskGroup
from app import create_app
from app.config import AppConfig
from loguru import logger

app = create_app(config=AppConfig)

@click.group(cls=FlaskGroup, create_app=lambda: app)
def cli():
    """Management script for the application."""

@cli.command('runserver')
@click.option('--debug', "debug", is_flag=True, help='Run the server in debug mode')
@click.option('--host', "host", default='0.0.0.0', help='Host to run the server on')
@click.option('--port', "port", default='5000', help='Port to run the server on')
def runserver(debug, host, port):
    if AppConfig.FLASK_ENV == 'development':
        logger.info(f"Running development server at {host}:{port}")
        os.environ['FLASK_RUN_HOST'] = host
        os.environ['FLASK_RUN_PORT'] = port
        os.environ['FLASK_ENV'] = AppConfig.FLASK_ENV
        os.environ['FLASK_DEBUG'] = '1' if debug else '0'
        cli(['run'])
    elif AppConfig.FLASK_ENV == 'production':
        logger.error("Production server not implemented yet, update config.env to development")


@app.shell_context_processor
def make_shell_context():
    return {'app': app}

if __name__ == '__main__':
    cli()