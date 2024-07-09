import os
import subprocess

import click
from flask.cli import FlaskGroup
from loguru import logger

from app import create_app
from app.config import AppConfig

app = create_app(config=AppConfig)


@click.group(cls=FlaskGroup, create_app=lambda: app)
def cli():
    """Management script for the application."""

@cli.command('hello')
def hello():
    """Prints hello world"""
    print('Hello World')

@cli.command('format')
def format():
    """Runs autoflake, yapf and isort formatters over the project"""
    autoflake_cmd = 'autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place *.py server/'
    isort_cmd = 'isort -rc *.py server/'
    yapf_cmd = 'yapf -r -i *.py server/'

    print('Running {}'.format(autoflake_cmd))
    subprocess.call(autoflake_cmd, shell=True)

    print('Running {}'.format(isort_cmd))
    subprocess.call(isort_cmd, shell=True)

    print('Running {}'.format(yapf_cmd))
    subprocess.call(yapf_cmd, shell=True)

@cli.command('train')
def train():
    """Runs the training script"""
    from app.handler import handle_train_model
    logger.debug("Training command called")
    handle_train_model()


@cli.command('runserver')
@click.option('--debug',
              "debug",
              is_flag=True,
              help='Run the server in debug mode')
@click.option('--host',
              "host",
              default='0.0.0.0',
              help='Host to run the server on')
@click.option('--port',
              "port",
              default='5000',
              help='Port to run the server on')
def runserver(debug, host, port):
    if AppConfig.FLASK_ENV == 'development':
        logger.info(f"Running development server at {host}:{port}")
        os.environ['FLASK_RUN_HOST'] = host
        os.environ['FLASK_RUN_PORT'] = port
        os.environ['FLASK_ENV'] = AppConfig.FLASK_ENV
        os.environ['FLASK_DEBUG'] = '1' if debug else '0'
        cli(['run'])
    elif AppConfig.FLASK_ENV == 'production':
        logger.error(
            "Production server not implemented yet, update config.env to development"
        )


@app.shell_context_processor
def make_shell_context():
    return {'app': app}


if __name__ == '__main__':
    cli()
