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


@cli.command('pipeline')
@click.option('--file', '-f', 'file', required=True, help='File of pipeline (the specific file in app/scripts/pipeline/...)')
@click.option('--method', '-m', 'method', required=True, help='Method of pipeline (the specific function in the category file)')
@click.option('--dataset', '-d', 'dataset', required=True, help='Dataset path to run the pipeline on')
@click.option('--config', '-c', 'config', help='Model configuration file for pipeline')
def pipeline(file, method, dataset, config):
    """Runs the pipeline script"""
    from app.handler import handle_pipeline
    logger.debug("Pipeline command called")
    result = handle_pipeline(file=file, method=method, dataset_path=dataset, model_config=config)
    logger.info(f"Pipeline command {'successful' if result else 'failed'}")

@cli.command('train')
def train():
    """Runs the training script"""
    from app.handler import handle_train_model
    logger.debug("Training command called")
    handle_train_model()
    
@cli.command('process')
@click.option('--input', '-i', 'input_path', required=True, help='Input path for Data')
@click.option('--output', '-o', 'output_path', required=True, help='Output path for Data')
@click.option('--file', '-f', 'file', required=True, help='File of processing (the specific file in app/scripts/process/...)')
@click.option('--method', '-m', 'method', required=True, help='Method of processing (the specific function in the category file)')
@click.option('--kwargs', '-k', multiple=True, help='Additional keyword arguments as key=value pairs')
def process(input_path, output_path, file, method, kwargs):
    """A Command line command that will process the data
    via the specified category file and method function and save the output to the output path

    Args:
        input_path (str): A path to the input data (absolute will be checked too)
        output_path (str): Output path to CropDoc/data, ensure to include the directory name
        category (str): The category of processing that coincides with the scripts/process
        method (str): The method of processing that coincides with the scripts/process
        kwargs (tuple): Additional keyword arguments to pass to the processing function
    """
    from app.handler import handle_process_data
    kwargs_dict = dict(kv.split('=') for kv in kwargs)
    result = handle_process_data(input_path, output_path, file, method, **kwargs_dict)
    logger.info(f"Processing {'successful' if result else 'failed'}")


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
