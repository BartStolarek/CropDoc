import os
import numpy as np
from PIL import Image
import cv2
from loguru import logger

def grayscale_using_cv2_recursive(input_directory, output_directory, **kwargs):
    """
    Recursively encode all images in a directory and its subdirectories in place.
    Rename images to include their directory path in the filename.

    Args:
        input_directory (str): Path to the directory containing images.
        output_directory (str): Path to the directory where processed images will be saved.
    """
    logger.debug(f"Encoding images recursively from {input_directory} to {output_directory}")

    # Ensure output directory exists; create if it doesn't
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    errors = []
    # Iterate through each file in the input directory
    for root, _, files in os.walk(input_directory):
        for filename in files:
            file_path = os.path.join(root, filename)

            # Check if file is an image (you can add more extensions as needed)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                try:
                    # Open the image using Pillow (PIL)
                    img = Image.open(file_path)

                    # Optionally, convert to numpy array and perform encoding using cv2
                    img_array = np.array(img)  # Convert image to numpy array

                    # Perform encoding operation (e.g., grayscale conversion using cv2)
                    encoded_img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

                    # Convert numpy array back to image
                    encoded_img = Image.fromarray(encoded_img_array)

                    # Extract directory path relative to the input directory
                    relative_path = os.path.relpath(root, input_directory)

                    # Replace directory separators with underscores
                    relative_path = relative_path.replace(os.sep, '_')

                    # Construct new filename with directory path and original filename
                    new_filename = f"{relative_path}_{filename}"
                    new_file_path = os.path.join(output_directory, new_filename)

                    # Save the encoded image, replacing the original image
                    encoded_img.save(new_file_path)

                    logger.debug(f"Encoded and saved: {file_path} -> {new_file_path}")

                except Exception as e:
                    logger.error(f"Error encoding {file_path}: {e}")
                    error_dict = {
                        "file": filename,
                        "error": str(e)
                    }
                    errors.append(error_dict)
                    
    if errors:
        logger.error("At least one file failed to be processed, run in debug mode to see the errors")
        logger.debug(errors)
        return False

    logger.info(f"Completed encoding images recursively")
    return True


