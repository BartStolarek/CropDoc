import traceback

from loguru import logger
import numpy as np
import gc
from typing import Tuple
import os
import tqdm
import shutil
from collections import Counter

from app.utility.file import get_file_method, load_yaml_file_as_dict
from app.pipeline.pipeline import Pipeline


def handle_pipeline(file: str,
                    method: str,
                    pipeline_config: str,
                    dataset: str = None,
                    **kwargs) -> bool:
    logger.debug(
        f"Handling pipeline with file {file} and method {method}, using dataset {dataset} and model config {pipeline_config}"
    )
    try:
        # Get the file method
        method_func = get_file_method(directory='pipeline',
                                      file_name=file,
                                      method_name=method,
                                      extension='py')

        # Load the model config
        pipeline_config = load_yaml_file_as_dict(directory='config',
                                                 file_name=pipeline_config)

        # Run pipeline command
        result = method_func(config=pipeline_config, **kwargs)

    except Exception as e:
        logger.error(
            f"Error running pipeline action: \n{traceback.print_exc()}\n{e}")
        return False

    logger.info("Pipeline completed successfully")

    return result


def handle_train(pipeline_config: str):
    """ A method/function to run the training pipeline

    Args:
        config (dict): A dictionary containing the configuration for the training pipeline

    Returns:
        _type_: _description_
    """
    logger.debug("Running the training pipeline")
    
    # Load the model config
    config = load_yaml_file_as_dict(directory='config', file_name=pipeline_config)

    # Initialise the Pipeline
    pipeline = Pipeline(config)

    # Train the modeol
    pipeline.train_model()

    # Save the model
    pipeline.save_pipeline()

    # Save Confusion matrix metrics
    pipeline.save_confusion_matrix()

    # Save the evaluation graphs
    pipeline.save_graphs()

    # Package the pipeline for API return
    pipeline_package = pipeline.package()

    logger.info("Training pipeline completed")

    return pipeline_package


def handle_predict(pipeline_config, image_path: str) -> dict:
    """ A method/function to run the prediction pipeline

    Args:
        config (dict): Config file that has been loaded as a dict
        image_path (str): A string pointing to the image in data/tmp

    Returns:
        dict: A dictionary containing the prediction results for the crop and state
    """

    logger.debug("Running the prediction pipeline")

    # Load the model config
    config = load_yaml_file_as_dict(directory='config', file_name=pipeline_config)
    
    # Initialise the Pipeline
    pipeline = Pipeline(config)

    # Predict the image
    prediction = pipeline.predict_model(image_path)

    # Save prediction to a file
    pipeline.save_prediction(prediction)

    logger.info("Prediction pipeline completed")

    return prediction


def handle_test(pipeline_config):
    """ A method/function to run the test pipeline

    Args:
        config (dict): A dictionary of all the configuration settings from config file

    Returns:
        _type_: _description_
    """
    logger.debug("Running the test pipeline")

    # Load the model config
    config = load_yaml_file_as_dict(directory='config', file_name=pipeline_config)
    
    # Initialise the Pipeline
    pipeline = Pipeline(config)

    # Test the model
    pipeline.test_model()
    
    # Save pipeline
    pipeline.save_pipeline()

    # Package the pipeline for API return
    pipeline_package = pipeline.package()

    logger.info("Test pipeline completed")

    return pipeline_package


def handle_predict_many(config, directory_path: str) -> Tuple[dict, dict]:
    """ A method/function to run the prediction pipeline on multiple images

    Args:
        config (dict): Dict containing the configuration settings from the config file
        directory_path (str): The path to the directory which contains the images to predict

    Returns:
        Tuple[dict, dict]: 
    """
    def calculate_evaluation(crop_data: dict, state_data: dict,
                             total_images: int) -> dict:
        evaluation = {
            'total_images': total_images,
            'crop': calculate_metrics(crop_data, total_images),
            'state': calculate_metrics(state_data, total_images)
        }
        return evaluation

    def calculate_metrics(data: dict, total_images: int) -> dict:
        metrics = {}
        for category in ['correct', 'incorrect']:
            probs = np.array(data[category]['probabilities'])
            confs = np.array(data[category]['confidences'])
            count = len(probs)

            metrics[category] = {
                'count': count,
                'probability': calculate_stats(probs),
                'confidence': calculate_stats(confs),
            }

            if category == 'incorrect':
                # Get the count of each incorrect prediction
                predictions = data[category]['predictions']
                prediction_count = Counter(predictions)
                metrics[category]['predictions'] = prediction_count

        return metrics

    def calculate_stats(arr: np.ndarray) -> dict:
        if len(arr) == 0:
            return {'average': 0, 'min': 0, 'max': 0, 'std': 0}
        return {
            'average': np.mean(arr),
            'min': np.min(arr),
            'max': np.max(arr),
            'std': np.std(arr)
        }

    logger.debug("Running the prediction pipeline on multiple images")

    directories = os.listdir(directory_path)
    logger.info(
        f"Found {len(directories)} directories in {directory_path}: {directories}"
    )

    evaluations = {}

    terminal_width = shutil.get_terminal_size().columns
    directory_progress = tqdm(directories,
                              desc="Directory",
                              leave=True,
                              ncols=int(terminal_width * 0.99))

    for directory in directory_progress:
        d_path = os.path.join(directory_path, directory)

        classes = directory.split('___')
        crop = classes[0]
        state = classes[1]

        if state == 'healthy':
            state = crop + '-healthy'

        # Get all image files in the directory
        #  image_files = [f for f in os.listdir(d_path) if os.path.isfile(os.path.join(d_path, f))]
        image_files = []
        count = 0
        for f in os.listdir(d_path):
            if count > 500:
                break
            if os.path.isfile(os.path.join(d_path, f)):
                image_files.append(f)
            else:
                logger.warning(f"Found a directory in {d_path}, skipping")
            count += 1

        logger.info(f"Found {len(image_files)} images in {d_path}")

        predictions = {}
        crop_data = {
            'correct': {
                'probabilities': [],
                'confidences': []
            },
            'incorrect': {
                'probabilities': [],
                'confidences': [],
                'predictions': []
            }
        }
        state_data = {
            'correct': {
                'probabilities': [],
                'confidences': []
            },
            'incorrect': {
                'probabilities': [],
                'confidences': [],
                'predictions': []
            }
        }

        prediction_progress = tqdm(image_files,
                                   desc="Image",
                                   leave=False,
                                   ncols=int(terminal_width * 0.99))

        for image_file in prediction_progress:

            image_path = os.path.join(d_path, image_file)
            prediction = handle_predict(config=config, image_path=image_path)

            # Garbage collect
            gc.collect()

            predictions[image_file] = prediction

            # Collect data for crop predictions
            if prediction['crop']['prediction'].lower() == crop.lower():
                crop_data['correct']['probabilities'].append(
                    prediction['crop']['probability'])
                crop_data['correct']['confidences'].append(
                    prediction['crop']['confidence'])
            else:
                crop_data['incorrect']['predictions'].append(
                    prediction['crop']['prediction'])
                crop_data['incorrect']['probabilities'].append(
                    prediction['crop']['probability'])
                crop_data['incorrect']['confidences'].append(
                    prediction['crop']['confidence'])

            # Collect data for state predictions
            if prediction['state']['prediction'].lower() == state.lower():
                state_data['correct']['probabilities'].append(
                    prediction['state']['probability'])
                state_data['correct']['confidences'].append(
                    prediction['state']['confidence'])
            else:
                state_data['incorrect']['probabilities'].append(
                    prediction['state']['probability'])
                state_data['incorrect']['confidences'].append(
                    prediction['state']['confidence'])
                state_data['incorrect']['predictions'].append(
                    prediction['state']['prediction'])

        evaluations[crop + '___' + state] = calculate_evaluation(
            crop_data, state_data, len(image_files))
        logger.info(f"Completed predictions for {crop}___{state}")

    return predictions, evaluations


def handle_evaluate(pipeline_config):
    """ A method/function to run the evaluation pipeline

    Args:
        config (dict): A dictionary of all the configuration settings from the config file

    Returns:
        _type_: _description_
    """
    logger.debug("Running the evaluation pipeline")

    # Load the model config
    config = load_yaml_file_as_dict(directory='config', file_name=pipeline_config)
    
    # Initialise the Pipeline
    pipeline = Pipeline(config)
    
    # Evaluate the model
    evaluation = pipeline.evaluate_model()
    
    