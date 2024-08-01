from app.config import AppConfig
from app.utility.path import join_path, file_exists, resolve_path
import importlib.util
from loguru import logger


def get_file_path(directory: str, file_name: str) -> str:
    """ Get the file path for the given directory and file name.

    Args:
        directory (str): directory that the file is in from the app/ directory
        file_name (str): the name of the file without the .py extension

    Returns:
        str: the file path if it exists, None otherwise
    """
    if file_name.endswith('.py'):
        file_name = file_name[:-3]
    file_path = resolve_path(join_path(AppConfig.APP_DIR + f"/{directory}/" + f"{file_name}.py"))
    if not file_exists(file_path):
        logger.error(f"File {file_name}.py does not exist in at {file_path}")
        return None
    return file_path


def import_file(file_path: str):
    """ Import a python file from a given file path.

    Args:
        file_path (str): the path to the python file

    Returns:
        _type_: the imported python file if successful, None otherwise
    """
    try:
        spec = importlib.util.spec_from_file_location("python_file", file_path)
        python_file = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(python_file)
    except Exception as e:
        logger.error(f"Error importing {file_path}: {e}")
        return False
    return python_file
    

def get_method(file, method_name: str):
    """ Get the method from a python file.

    Args:
        file (_type_): the imported python file
        method_name (str): the name of the method to get

    Returns:
        _type_: the method if it exists, None otherwise
    """
    if not hasattr(file, method_name):
        logger.error(f"Method {method_name} does not exist in {file}.py")
        return None
    return getattr(file, method_name)


def get_file_method(directory: str, file_name: str, method_name: str):
    """ Get the method from a python file in a given directory.

    Args:
        directory (str): The directory that the file is in from the app/ directory
        file_name (str): The name of the file without the .py extension
        method_name (str): The name of the method to get

    Returns:
        _type_: the method if it exists, None otherwise
    """
    file_path = get_file_path(directory, file_name)
    if not file_path:
        return None
    file = import_file(file_path)
    if not file:
        return None
    method = get_method(file, method_name)
    if not method:
        return None
    return method



