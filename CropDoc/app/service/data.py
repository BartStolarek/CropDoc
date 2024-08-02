import os
import shutil
from loguru import logger
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose
from app.config import DATA_DIR
from app.utility.path import directory_exists
from torch.utils.data import ConcatDataset
from typing import Union
import numpy as np


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


class ResNet50V2DatasetManager():
    def __init__(self, root_path: str, transform: Compose):
        self.dataset = ImageFolder(root_path, transform=transform)
        self.name = self.dataset.root.split("/")[-1]
        self.class_map = self.map_dataset_to_dict()
        self.class_count = self.get_class_count()
        self.sample_count = self.get_sample_count()
        logger.info(f"Dataset {self.name} loaded with {self.class_count} classes and {self.sample_count} samples")
    
    def __str__(self):
        return f"{self.name} Dataset Manager"
    
    def get_class_count(self) -> int:
        """ Get the count of classes for the dataset.

        Returns:
            int: The count of classes for the dataset.
        """
        logger.debug(f"Getting class count for {self.name}")
        count = len(self.class_map)
        logger.info(f"Class count for dataset {self.name}: {count}")
        return count
      
    def map_dataset_to_dict(self) -> dict:
        """ Map an ImageFolder dataset to a dictionary of class counts.


        Raises:
            ValueError: If no dataset is provided.

        Returns:
            dict: A dictionary of class counts for the dataset.
        """
        logger.debug(f"Mapping dataset to dictionary: {self.name}")
        try:
            
            # Get a list of classes
            class_names = self.dataset.classes
            
            # Get a array of counts
            class_counts = np.bincount(self.dataset.targets)
            
            # Create a dictionary of class names and counts
            class_count_dict = {class_: count for class_, count in zip(class_names, class_counts)}
            
            # Return the dictionary
            logger.info(f"Class count dictionary for {self.name}: {class_count_dict}")
            return class_count_dict

        except Exception as e:
            logger.error(f"Error mapping dataset to dictionary: {e}")
            return {}
    
    def get_sample_count(self) -> int:
        """ Get the sample count for the dataset.

        Raises:
            ValueError: If the sample count does not match the class count.

        Returns:
            int: The sample count for the dataset.
        """
        logger.debug(f"Getting sample count for {self.name}")
        count = sum([value for key, value in self.class_map.items])
        if count != len(self.dataset):
            logger.error(f"Sample count does not match class count: {count} != {len(self.dataset)}")
            raise ValueError(f"Sample count does not match class count: {count} != {len(self.dataset)}")
        logger.info(f"Sample count for dataset {self.name}: {count}")
        return count


class ResNet50V2ConcatDatasetManager():
    """ Class for managing multiple datasets as a single dataset.
    """
    def __init__(self, datasets: list[ResNet50V2DatasetManager], name: str):

        if not isinstance(datasets, list[ResNet50V2DatasetManager]):
            logger.error("Datasets must be a list of DatasetManager objects")
            raise ValueError("Datasets must be a list of DatasetManager objects")
        
        self.datasets = datasets
        self.concat_dataset = ConcatDataset(datasets)
        self.name = name
        self.class_map = self.map_datasets_to_dict()
        self.level_counts = self.get_class_count_per_level()
        self.sample_count = self.get_sample_count()
        
    def map_datasets_to_dict(self) -> dict:
        """ Map a list of datasets to a dictionary of class counts.

        Returns:
            dict: A dictionary of class counts for each dataset in the list.
        """
        logger.debug(f"Mapping datasets to dictionary: {self.name}")
        try:
            class_map = {}
            for dataset in self.datasets:
                class_map[dataset.name] = dataset.class_map
            
            logger.info(f"Class map for {self.name}: {class_map}")
            return class_map
        
        except Exception as e:
            logger.error(f"Error mapping datasets to dictionary: {e}")
            return {}
        
    def get_class_count_per_level(self, dictionary: dict = None, level=1, level_counts=None) -> dict:
        """ Get the count of classes per level in the dataset.

        Args:
            dictionary (dict, optional): The dictionary to count the classes of. Defaults to None.
            level (int, optional): The level of the dictionary. Defaults to 1.
            level_counts (_type_, optional): The dictionary to store the counts in. Defaults to None.

        Returns:
            dict: A dictionary of class counts per level.
        """
        try:
            logger.debug(f"Getting class count per level for {self.name}")
            if level_counts is None:
                level_counts = {}
                
            if dictionary is None:
                dictionary = self.class_map

            for key, value in dictionary.items():
                if isinstance(value, dict):
                    self.get_class_count_per_level(value, level + 1, level_counts)
                else:
                    if level in level_counts:
                        level_counts[level] += 1
                    else:
                        level_counts[level] = 1
            logger.info(f"Class count per level for {self.name}: {level_counts}")
            return level_counts
        except Exception as e:
            logger.error(f"Error getting class count per level: {e}")
            return {}
    
    def get_sample_count(self) -> int:
        """ Get the sample count for the dataset.

        Raises:
            ValueError: If the sample count does not match the class count.

        Returns:
            int: The sample count for the dataset.
        """
        logger.debug(f"Getting sample count for {self.name}")
        try:
            count = sum([dataset.get_sample_count() for dataset in self.datasets])
            if count != len(self.datasets):
                logger.error(f"Sample count does not match class count: {count} != {len(self.datasets)}")
                raise ValueError(f"Sample count does not match class count: {count} != {len(self.datasets)}")
            logger.info(f"Sample count for dataset {self.name}: {count}")
            return count
        except Exception as e:
            logger.error(f"Error getting sample count: {e}")
            return 0
