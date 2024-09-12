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
    autoflake_cmd = 'autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place *.py app/'
    isort_cmd = 'isort -rc *.py app/'
    yapf_cmd = 'yapf -r -i *.py app/'

    print('Running {}'.format(autoflake_cmd))
    subprocess.call(autoflake_cmd, shell=True)

    print('Running {}'.format(isort_cmd))
    subprocess.call(isort_cmd, shell=True)

    print('Running {}'.format(yapf_cmd))
    subprocess.call(yapf_cmd, shell=True)


@cli.command('pipeline')
@click.option(
    '--file',
    '-f',
    'file',
    required=True,
    help='File of pipeline (the specific file in app/scripts/pipeline/...)')
@click.option(
    '--method',
    '-m',
    'method',
    required=True,
    help='Method of pipeline (the specific function in the category file)')
@click.option('--config',
              '-c',
              'config',
              required=True,
              help='Model configuration file for pipeline')
@click.option('--dataset',
              '-d',
              'dataset',
              default=None,
              help='Dataset path to run the pipeline on')
@click.option('--kwargs',
              '-k',
              multiple=True,
              help='Additional keyword arguments as key=value pairs')
def pipeline(file, method, config, dataset, kwargs):
    """Runs the pipeline script"""
    from app.handler import handle_pipeline
    logger.debug("Pipeline command called")
    kwargs_dict = dict(kv.split('=') for kv in kwargs)
    result = handle_pipeline(file=file,
                             method=method,
                             dataset=dataset,
                             pipeline_config=config,
                             **kwargs_dict)
    logger.info(f"Pipeline command {'successful' if result else 'failed'}")
    
@cli.command('train')
@click.option('--config',
              '-c',
              'config',
              required=True,
              help='Model configuration file for pipeline')
@click.option('--kwargs',
              '-k',
              multiple=True,
              help='Additional keyword arguments as key=value pairs')
def train(config, kwargs):
    """Runs the train pipeline script"""
    from app.handler import handle_train
    logger.debug("Pipeline command called")
    kwargs_dict = dict(kv.split('=') for kv in kwargs)
    result = handle_train(pipeline_config=config, **kwargs_dict)
    logger.info(f"Pipeline train command {'successful' if result else 'failed'}")
    
@cli.command('predict')
@click.option('--config',
              '-c',
              'config',
              required=True,
              help='Model configuration file for pipeline')
@click.option('--image',
              '-i',
              'image',
              required=True,
              help='Image path to run the pipeline on')
@click.option('--kwargs',
              '-k',
              multiple=True,
              help='Additional keyword arguments as key=value pairs')
def predict(config, image, kwargs):
    """Runs the predict pipeline script"""
    from app.handler import handle_predict
    logger.debug("Pipeline command called")
    kwargs_dict = dict(kv.split('=') for kv in kwargs)
    result = handle_predict(pipeline_config=config, image_path=image, **kwargs_dict)
    logger.info(f"Pipeline predict command {'successful' if result else 'failed'}")
    
@cli.command('test')
@click.option('--config',
              '-c',
              'config',
              required=True,
              help='Model configuration file for pipeline')
@click.option('--kwargs',
              '-k',
              multiple=True,
              help='Additional keyword arguments as key=value pairs')
def test(config, kwargs):
    """Runs the test pipeline script"""
    from app.handler import handle_test
    logger.debug("Pipeline command called")
    kwargs_dict = dict(kv.split('=') for kv in kwargs)
    result = handle_test(pipeline_config=config, **kwargs_dict)
    logger.info(f"Pipeline test command {'successful' if result else 'failed'}")

 
@cli.command('predict_many')
@click.option('--config',
              '-c',
              'config',
              required=True,
              help='Model configuration file for pipeline')
@click.option('--kwargs',
              '-k',
              multiple=True,
              help='Additional keyword arguments as key=value pairs')
def predict_many(config, kwargs):
    """Runs the predict_many pipeline script"""
    from app.handler import handle_predict_many
    logger.debug("Pipeline command called")
    kwargs_dict = dict(kv.split('=') for kv in kwargs)
    result = handle_predict_many(pipeline_config=config, **kwargs_dict)
    logger.info(f"Pipeline predict_many command {'successful' if result else 'failed'}")


@cli.command('evaluate')
@click.option('--config',
              '-c',
              'config',
              required=True,
              help='Model configuration file for pipeline')
@click.option('--kwargs',
              '-k',
              multiple=True,
              help='Additional keyword arguments as key=value pairs')
def evaluate(config, kwargs):
    """Runs the evaluate pipeline script"""
    from app.handler import handle_evaluate
    logger.debug("Pipeline command called")
    kwargs_dict = dict(kv.split('=') for kv in kwargs)
    result = handle_evaluate(pipeline_config=config, **kwargs_dict)
    logger.info(f"Pipeline evaluate command {'successful' if result else 'failed'}")



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
@click.option('--reload',
              "reload",
              is_flag=True,
              help='Reload the server on file changes')
def runserver(debug, host, port, reload):
    if AppConfig.FLASK_ENV == 'development':
        logger.info(f"Running development server at {host}:{port}")
        os.environ['FLASK_RUN_HOST'] = host
        os.environ['FLASK_RUN_PORT'] = port
        os.environ['FLASK_ENV'] = AppConfig.FLASK_ENV
        os.environ['FLASK_DEBUG'] = '1' if debug else '0'
        os.environ['FLASK_RUN_RELOAD'] = '1' if reload else '0'
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
