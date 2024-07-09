from loguru import logger
import os
from app.config import AppConfig
from app.utility.path import directory_exists, join_path, file_exists, directory_is_empty, resolve_path, combine_path
from app.service.data import import_images_in_flat_structure
import importlib.util


def handle_process_data(input_path: str, output_path: str, file: str, method: str, **kwargs) -> bool:
    logger.debug(f"Processing data from {input_path} to {output_path} using {file} (app/scripts/process/file) and {method} (function) with {kwargs}")
    
    # Check path exists
    if not directory_exists(input_path):
        logger.error(f"Input path {input_path} does not exist or isn't a directory")
        return False
    
    # Check if output path exists, if not create one in CropDoc/data
    output_path = combine_path(AppConfig.DATA_DIR, output_path)
    if directory_exists(output_path):
        if not directory_is_empty(output_path):
            logger.error(f"Output path {output_path} already exists and is not empty, please provide a new directory")
            return False
    else:
        os.makedirs(output_path)
        
    # Check if the file exists in app/scripts/process
    process_file_path = resolve_path(join_path(AppConfig.APP_DIR + '/scripts/process/' + f"{file}.py"))
    if not file_exists(process_file_path):
        logger.error(f"File {file}.py does not exist in at {process_file_path}")
        return False

    # Import the file
    try:
        spec = importlib.util.spec_from_file_location("process_file", process_file_path)
        process_file = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(process_file)
    except Exception as e:
        logger.error(f"Error importing {process_file_path}: {e}")
        return False
    
    # Check if the method (function) exists in the file
    if not hasattr(process_file, method):
        logger.error(f"Method {method} does not exist in {file}.py")
        return False
    
    # Get the method (function)
    method_func = getattr(process_file, method)
        

    # Process the data
    try:
        result = method_func(input_path, output_path, **kwargs)
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        return False
    
    if result:
        logger.info("Successfully processed data")
    else:
        logger.error("Failed to process all data")
    
    return True
