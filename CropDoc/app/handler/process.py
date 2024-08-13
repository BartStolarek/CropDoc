import os

from loguru import logger

from app.config import AppConfig
from app.utility.file import get_file_method
from app.utility.path import combine_path, directory_exists, directory_is_empty


def handle_process_data(input_path: str, output_path: str, file: str,
                        method: str, **kwargs) -> bool:
    logger.debug(
        f"Processing data from {input_path} to {output_path} using {file} (app/process/file) and {method} (function) with {kwargs}"
    )

    # Check path exists
    if not directory_exists(input_path):
        logger.error(
            f"Input path {input_path} does not exist or isn't a directory")
        return False

    # Check if output path exists, if not create one in CropDoc/data
    output_path = combine_path(AppConfig.DATA_DIR, output_path)
    if directory_exists(output_path):
        if not directory_is_empty(output_path):
            logger.error(
                f"Output path {output_path} already exists and is not empty, please provide a new directory"
            )
            return False
    else:
        os.makedirs(output_path)

    # Get the file method
    method_func = get_file_method(directory='process',
                                  file_name=file,
                                  method=method)

    # Process the data
    try:
        result = method_func(input_path, output_path, **kwargs)
    except TypeError as e:
        logger.error(
            f"Potentially incorrect arguments in {method} function, process handler provides a 'input_path', 'output_path' and **kwargs, your process function should at least have those 3: {e}"
        )
        return False
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        return False

    if result:
        logger.info("Successfully processed data")
    else:
        logger.error("Failed to process all data")

    return True
