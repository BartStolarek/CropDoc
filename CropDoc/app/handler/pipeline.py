from loguru import logger
import os
from app.config import AppConfig
from app.utility.path import directory_exists, directory_is_empty, join_path
from app.utility.file import get_file_method, load_yaml_file_as_dict
from app.service.data import DatasetManager
import traceback


def handle_pipeline(file: str, method: str, dataset_path: str, model_config: str) -> bool:
    logger.debug(f"Handling pipeline with file {file} and method {method}")
    try:    
        # Get the file method
        method_func = get_file_method(directory='pipeline', file_name=file, method_name=method, extension='py')     

        # Load the model config
        model_config = load_yaml_file_as_dict(directory='config', file_name=model_config)

        # Run pipeline command
        method_func(dataset_path=dataset_path, config=model_config)
        
    except Exception as e:
        logger.error(f"Error running pipeline action: \n{traceback.print_exc()}\n{e}")
        return False
    

    logger.info("Pipeline completed successfully")
    
    return True
