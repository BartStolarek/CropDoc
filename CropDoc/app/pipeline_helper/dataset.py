import os
import re
from abc import ABC, abstractmethod

import numpy as np
import torch
from loguru import logger
from PIL import Image


class BaseDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, root: str, existing_structure: dict = None, test_split: float = None):
        self.root = root
        
        # If existing structure was provided, use that structure
        if existing_structure is not None:
            self.train_images = existing_structure['train_images']
            self.train_labels = existing_structure['train_labels']
            self.test_images = existing_structure['test_images']
            self.test_labels = existing_structure['test_labels']
            self.crops = existing_structure['crops']
            self.states = existing_structure['states']
            self.train_map = existing_structure['train_map']
            self.test_map = existing_structure['test_map']
        else:
            train_images, train_labels, test_images, test_labels = self.structure_data(test_split)
            
            # Check if train and test images and labels are numpy arrays
            if not isinstance(train_images, np.ndarray) or not isinstance(test_images, np.ndarray) or not isinstance(train_labels, np.ndarray) or not isinstance(test_labels, np.ndarray):
                raise ValueError("structure_data() must numpy arrays for train images, test images, train labels, and test labels")
            else:
                self.train_images = train_images
                self.test_images = test_images
                
                # Check if train and test labels are tuples of crop and state labels
                if len(train_labels[0]) != 2 or len(test_labels[0]) != 2:
                    print(len(test_labels[0]))
                    raise ValueError(f"structure_data() must return numpy array for train and test labels, which are tuples of crop and state labels, currently returning {train_labels[0]}")
                else:
                    self.train_labels = train_labels
                    self.test_labels = test_labels
            
            # Generate the unique index maps for crops and states if not provided, same across both train and test splits
            self.crops = self.get_crops() 
            self.states = self.get_states()
            
            # Generate the count maps for train and test splits
            self.train_map = self.generate_map('train')
            self.test_map = self.generate_map('test')
        
    def generate_map(self, split) -> dict:
        """ Method to generate a map of the dataset which shows the count
        of each crop class and state class in the dataset

        Returns:
            dict: _description_
        """
        if split == 'train':
            labels = self.train_labels
        elif split == 'test':
            labels = self.test_labels
        
        data_map = {
            'crop': {},
            'state': {}
        }
        for crop, state in labels:
            if crop not in data_map['crop']:
                data_map['crop'][crop] = 1
            else:
                data_map['crop'][crop] += 1
                
            if state not in data_map['state']:
                data_map['state'][state] = 1
            else:
                data_map['state'][state] += 1
        return data_map
                
    def get_crops(self) -> np.ndarray:
        """ Method to get a unique numpy array (list) of crops from the dataset

        Returns:
            np.ndarray: A numpy array of unique crops
        """
        train_crops = np.unique(self.train_labels[:, 0])
        test_crops = np.unique(self.test_labels[:, 0])
        
        # Check if the train and test crops have exact same crops
        if not np.array_equal(train_crops, test_crops):
            raise ValueError(f"Train and test crops must be the same, there are different crops in the train and test splits: \nTrain Crops:\n{train_crops}, \nTest Crops:\n{test_crops}")
        
        logger.info(f'Generated new crop index: {train_crops}')
        for crop in train_crops:
            print(crop)
        return train_crops

    def get_states(self) -> np.ndarray:
        """ Method to get a unique numpy array (list) of states from the dataset

        Returns:
            np.ndarray: A numpy array of unique states
        """
        train_states = np.unique(self.train_labels[:, 1])
        test_states = np.unique(self.test_labels[:, 1])
        
        # Check if the train and test crops have exact same crops
        if not np.array_equal(train_states, test_states):
            raise ValueError(f"Train and test crops must be the same, there are different crops in the train and test splits: \nTrain Crops:\n{train_states}, \nTest Crops:\n{test_states}")
        
        logger.info(f'Generated new state index: {train_states}')
        for state in train_states:
            print(state)
        return train_states
          
    @abstractmethod
    def structure_data(self, test_split: float = None):
        """ Method that returns the following:
        - A numpy array of image paths
        - A numpy array of tuples of crop and state labels (cleaned), where index corresponds to image path
        - A numpy array of splits (train, val, test) for each image path
        """
        pass
        
    def __getitem__(self, idx):
        """_summary_

        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        img_path = self.images[idx]
        crop, state = self.labels[idx]

        img = Image.open(img_path).convert('RGB')

        # Convert crop and state to numeric labels
        crop_label = self.crops.index(crop)
        state_label = self.states.index(state)

        return img, crop_label, state_label

    def __len__(self) -> int:
        """ Method to return the length of the dataset

        Returns:
            int: The length of the dataset
        """
        return len(self.images)
    
    def combine(self, other_dataset):
        """
        Combine this dataset with another dataset that is a subclass of BaseDataset.
        
        Args:
            other_dataset (BaseDataset): Another dataset to combine with this one.
        """
        if not issubclass(type(other_dataset), type(self)):
            raise TypeError("Can only combine datasets of the same type or its subclasses")
        
        # Combine images
        self.images = np.concatenate((self.images, other_dataset.images))
        
        # Combine labels
        self.labels = np.concatenate((self.labels, other_dataset.labels))
        
        # Update crops and states
        self.crops = np.unique(np.concatenate((self.crops, other_dataset.crops)))
        self.states = np.unique(np.concatenate((self.states, other_dataset.states)))
        
        # Update the map
        self.map = self.generate_map()

   
class CropCCMTDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def structure_data(self, test_split: float = None):
        
        train_images = []
        train_labels = []
        test_images = []
        test_labels = []
        
        # Get each directory name in the root directory
        crop_directories = os.listdir(self.root)
        
        for crop in crop_directories:
            
            split_directory = os.listdir(os.path.join(self.root, crop))
            
            for split in split_directory:
                
                state_directories = os.listdir(os.path.join(self.root, crop, split))
                
                for state in state_directories:
                    
                    image_files = os.listdir(os.path.join(self.root, crop, split, state))
                    
                    for image_file in image_files:
                        
                        if not image_file.lower().endswith('.jpg') and not image_file.lower().endswith('.jpeg'):
                            logger.debug(f'File {image_file} is not a jpg file, skipping')
                            continue
                        
                        image_path = os.path.join(self.root, crop, split, state, image_file)
                        
                        crop_label = crop.title()
                        
                        state_label = crop_label + '-Healthy' if 'healthy' in state.lower() else state.title()
                        
                        if 'test' in split:
                            test_images.append(image_path)
                            test_labels.append((crop_label, state_label))
                        elif 'train' in split:
                            train_images.append(image_path)
                            train_labels.append((crop_label, state_label))
        
        train_images = np.array(train_images, dtype=object)
        train_labels = np.array(train_labels, dtype=object)
        test_images = np.array(test_images, dtype=object)
        test_labels = np.array(test_labels, dtype=object)
        
        logger.info(f'Finished structuring data, found {len(train_images)} train images and {len(test_images)} test images')
        
        return train_images, train_labels, test_images, test_labels

                         
class PlantVillageDataset(BaseDataset):
    """A class to load the dataset from the data/dataset directory
    
    Dataset can be found: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset?resource=download

    Args:
        BaseDataset (torch.utils.data.Dataset): Base class for all datasets in PyTorch
    """

    def __init__(self, test_split: float, **kwargs):
        super().__init__(test_split=test_split, **kwargs)
        
    def structure_data(self, test_split: float = None):
        # Go through the dataset directory and get all the images and labels
        # Then use the test_split to split the data into train and test splits, return those
        if test_split is None:
            raise ValueError("test_split must be provided for PlantVillageDataset")
        
        directories = os.listdir(self.root)
        
        image_paths = []
        labels = []
        
        for directory in directories:
            labels = directory.split('___')
            crop = labels[0].title()
            state = labels[1].title()
            
            if ',' in crop:
                crop = crop.replace(',', '')
                
            if '_' in crop:
                crop = crop.replace('_', ' ')
                crop.title()
                
            if '_' in state:
                state = state.replace('_', ' ')
                state.title()
            
            if state == 'Healthy':
                state = crop + '-Healthy'
            
            if 'tomato mosaic virus' in state.lower():
                state = 'Mosaic'
            
            if 'cercospora leaf spot gray leaf spot' in state.lower():
                state = 'Leaf Spot'
            
            if 'leaf blight (isariopsis leaf spot)' in state.lower():
                state = 'Leaf Blight'
                
            if 'northern leaf blight' in state.lower():
                state = 'Leaf Blight'
                
            if 'early blight' in state.lower():
                state = 'Leaf Blight'
            
            if 'late blight' in state.lower():
                state = 'Leaf Blight'
                
            if 'bacterial spot' in state.lower():
                state = 'Bacterial Spot'
            
            
            for image in os.listdir(os.path.join(self.root, directory)):
                if not image.lower().endswith('.jpg') and not image.lower().endswith('.jpeg'):
                    logger.debug(f'File {image} is not a jpg file, skipping')
                    continue
                
                image_path = os.path.join(self.root, directory, image)
                image_paths.append(image_path)
                labels.append((crop, state))
                
            
        crop_labels = np.unique(np.array(crop_labels))
        state_labels = np.unique(np.array(state_labels))
        

