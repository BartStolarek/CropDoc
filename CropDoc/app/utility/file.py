import importlib.util

import yaml
from loguru import logger

from app.config import AppConfig
from app.utility.path import file_exists, join_path, resolve_path


def get_file_path(directory: str,
                  file_name: str,
                  extension: str = None) -> str:
    """ Get the file path for the given directory and file name.

    Args:
        directory (str): directory that the file is in from the app/ directory
        file_name (str): the name of the file without the .py extension

    Returns:
        str: the file path if it exists, None otherwise
    """
    logger.debug(f"Getting file path for {file_name}.py in {directory}")
    # If file path has extension, remove it
    if '.' not in file_name:
        if extension:
            file_name = f"{file_name}.{extension}"
        else:
            logger.error(
                f"No extension provided via command line or argument for {file_name}"
            )
    file_path = resolve_path(
        join_path(AppConfig.APP_DIR + f"/{directory}/" + f"{file_name}"))
    if not file_exists(file_path):
        logger.error(f"File {file_name} does not exist in at {file_path}")
        return None
    logger.info(f"Obtained file path for {file_name} in {directory}")
    return file_path


def import_file(file_path: str):
    """ Import a python file from a given file path.

    Args:
        file_path (str): the path to the python file

    Returns:
        _type_: the imported python file if successful, None otherwise
    """
    logger.debug(f"Importing python file from {file_path}")
    try:
        spec = importlib.util.spec_from_file_location("python_file", file_path)
        python_file = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(python_file)
    except Exception as e:
        logger.error(f"Error importing {file_path}: {e}")
        return False
    logger.info(f"Successfully imported python file from {file_path}")
    return python_file


def get_method(file, method_name: str):
    """ Get the method from a python file.

    Args:
        file (_type_): the imported python file
        method_name (str): the name of the method to get

    Returns:
        _type_: the method if it exists, None otherwise
    """
    logger.debug(f"Getting method {method_name} from {file}")
    if not hasattr(file, method_name):
        logger.error(f"Method {method_name} does not exist in {file}.py")
        return None
    logger.info(f"Obtained method {method_name} from {file}")
    return getattr(file, method_name)


def get_file_method(directory: str,
                    file_name: str,
                    method_name: str,
                    extension: str = None):
    """ Get the method from a python file in a given directory.

    Args:
        directory (str): The directory that the file is in from the app/ directory
        file_name (str): The name of the file without the .py extension
        method_name (str): The name of the method to get

    Returns:
        _type_: the method if it exists, None otherwise
    """
    logger.debug(
        f"Getting method {method_name} from {file_name}.{extension} in {directory}"
    )
    file_path = get_file_path(directory, file_name, extension=extension)
    if not file_path:
        return None
    file = import_file(file_path)
    if not file:
        return None
    method = get_method(file, method_name)
    if not method:
        return None
    logger.info(
        f"Obtained method {method_name} from {file_name}.py in {directory}")
    return method


def load_yaml_file_as_dict(directory: str, file_name: str) -> dict:
    """ Load a YAML model configuration file from the app/model directory.

    Args:
        file_name (str): The name of the YAML file without the .yml extension

    Returns:
        dict: The parsed contents of the YAML file as a dictionary
    """
    logger.debug(f"Loading model configuration file {file_name}.yml")
    file_path = get_file_path(directory, file_name, extension='yml')
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(
            f"Successfully loaded model configuration file {file_name}.yml")
        return config
    except FileNotFoundError:
        logger.error(
            f"Model configuration file {file_name}.yml not found at {file_path}"
        )
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {file_name}.yml: {e}")
    return None
