import os
import shutil
from loguru import logger


def import_images_in_flat_structure(input_directory, output_directory) -> bool:
    """
    Imports all images from input_directory and saves them in output_directory.
    Each image will be renamed to include its directory path in the filename.

    Args:
        input_directory (str): Path to the directory containing images.
        output_directory (str): Path to the directory where processed images will be saved.
    
    Returns:
        bool: True if all successful, False otherwise.
    """
    logger.debug(f"Importing images from {input_directory} to {output_directory}")
    # Ensure output directory exists; create if it doesn't
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    errors = []
    # Iterate through each file in input_directory
    for root, _, files in os.walk(input_directory):
        for file in files:
            # Check if file is an image (you can add more extensions as needed)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                try:
                    file_path = os.path.join(root, file)
                    
                    # Extract directory path relative to input_directory
                    relative_path = os.path.relpath(root, input_directory)
                    
                    # Replace directory separators with underscores
                    relative_path = relative_path.replace(os.sep, '_')
                    
                    # Create output filename with directory path and original filename
                    output_filename = os.path.join(output_directory, f"{relative_path}_{file}")
                    
                    # Copy the file to the output directory
                    shutil.copy(file_path, output_filename)
                    
                    logger.debug(f"Copied: {file_path} -> {output_filename}")
                except Exception as e:
                    error_dict = {
                        "file": file,
                        "error": str(e)
                    }
                    errors.append(error_dict)
    
    if errors:
        logger.error("At least one file failed to be processed, run in debug mode to see the errors")
        logger.debug(errors)
        return False
                             
    logger.info(f"Completed importing images")
    return True
