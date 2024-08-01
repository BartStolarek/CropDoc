import os
import shutil
from loguru import logger
from torchvision.datasets import ImageFolder
from app.config import DATA_DIR
from app.utility.path import directory_exists


def load_crop_ccmt_dataset_augmented(crops: list, transformers: dict) -> tuple:
    """ Load the CCMT Dataset-Augmented for the specified crops and transformers.

    Args:
        crops (list): A list of crops to load the dataset for.
        transformers (dict): A dictionary of torchvision.transformers.Compose with transformers within for the training and test sets.

    Raises:
        ValueError: If the transformers dictionary does not have keys 'train' and 'test'.
        FileNotFoundError: If the dataset directory does not exist.

    Returns:
        tuple: A tuple containing the training datasets, test datasets and all classes for each crop.
    """
    if transformers.keys() != ['train', 'test']:
        logger.error("Transformers must be a dictionary with keys 'train' and 'test'")
        raise ValueError("Transformers must be a dictionary with keys 'train' and 'test'")
    
    dataset_directory = os.path.join(DATA_DIR, 'datasets', 'CCMT Dataset-Augmented')
    
    if not directory_exists(dataset_directory):
        logger.error(f"Dataset directory {dataset_directory} does not exist")
        raise FileNotFoundError(f"Dataset directory {dataset_directory} does not exist")
    
    train_datasets = []
    test_datasets = []
    all_classes = {}
    
    # For each crop, load the dataset
    for crop in crops:
        crop_dir = os.path.join(DATA_DIR, 'datasets', 'CCMT Dataset-Augmented', crop)
        
        # Get the crops training set
        train_dataset = ImageFolder(os.path.join(crop_dir, 'train_set'), transform=transformers['train'])
        
        # Get the crops test set
        test_dataset = ImageFolder(os.path.join(crop_dir, 'test_set'), transform=transformers['test'])
        
        # Append the training and test datasets to the lists
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)
        
        # Check that all train dataset classes are the same as the test dataset classes
        assert train_dataset.classes == test_dataset.classes, f"Train and test classes do not match for {crop}"
        
        crop_classes = train_dataset.classes
        all_classes[crop] = crop_classes
        
    return train_datasets, test_datasets, all_classes
 

def count_all_classes(all_classes: dict) -> int:
    """ Count the total number of classes in all crops.

    Args:
        all_classes (dict): A dictionary containing all classes for each crop.

    Returns:
        int: The total number of classes in all crops.
    """
    try:
        class_count_per_crop = []
        for crop, classes in all_classes.items():
            class_count_per_crop.append(len(classes))
            
        return sum(class_count_per_crop)
    except Exception as e:
        logger.error(f"Error counting all classes: {e}")
        return 0
    
def get_unique_class_count(all_classes: dict) -> dict:
    """ Get the unique class count for each crop.

    Args:
        all_classes (dict): A dictionary containing all classes for each crop.

    Returns:
        dict: A dictionary containing the unique class count for each crop.
    """
    try:
        unique_class_count = {}
        for crop, classes in all_classes.items():
            for class_ in classes:
                if class_ not in unique_class_count:
                    unique_class_count[class_] = 0
                unique_class_count[class_] += 1
    except Exception as e:
        logger.error(f"Error getting unique class count: {e}")
        return {}
    