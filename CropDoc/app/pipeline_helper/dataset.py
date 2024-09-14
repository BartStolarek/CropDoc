import os
import re
from abc import ABC, abstractmethod

import numpy as np
import torch
from loguru import logger
from PIL import Image
import json

class Structure():
    def __init__(self, train_images=None, train_labels=None, test_images=None, test_labels=None, crops=None, states=None):
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels
        self.crops = crops
        self.states = states
        
    def save_structure(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        save_dict = self.convert_to_dict()
        
        with open(os.path.join(directory, 'structure.json'), 'w') as f:
            json.dump(save_dict, f)
    
    def convert_to_dict(self):
        new_dict = {}
        
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                new_dict[key] = value.tolist()
            else:
                new_dict[key] = value
        return new_dict
     
    def load_dict(self, directory):
        file_path = os.path.join(directory, 'structure.json')
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            return None
    
    def load_structure(self, directory):
        loaded_dict = self.load_dict(directory)
        if not loaded_dict:
            return None
        self.train_images = np.array(loaded_dict['train_images'])
        self.train_labels = np.array(loaded_dict['train_labels'])
        self.test_images = np.array(loaded_dict['test_images'])
        self.test_labels = np.array(loaded_dict['test_labels'])
        self.crops = np.array(loaded_dict['crops'])
        self.states = np.array(loaded_dict['states'])
        
    def __str__(self):
        string = f"Train Images: {len(self.train_images)}, " + \
            f"Test Images: {len(self.test_images)}, " + \
            f"Crops: {len(self.crops)}, " + \
            f"States: {len(self.states)}"
        return string
    
    @staticmethod
    def array_equal_unordered(a, b):
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        return np.array_equal(np.sort(a), np.sort(b))


class BaseDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, root: str, name: str, test_split: float = None):
        self.roots = [root]
        self.ids = [name]
        # If existing structure was provided, use that structure
        
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
    
    def extract_structure(self):
        structure = Structure(
            train_images=self.train_images,
            train_labels=self.train_labels,
            test_images=self.test_images,
            test_labels=self.test_labels,
            crops=self.crops,
            states=self.states
        )
        return structure
      
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
        
        logger.info('Generated new crop index')
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
        
        logger.info('Generated new state index')
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
        return len(self.train_images)
    
    def combine(self, other_dataset):
        """
        Combine this dataset with another dataset that is a subclass of BaseDataset.
        
        Args:
            other_dataset (BaseDataset): Another dataset to combine with this one.
        """
        # If parent class of other_dataset is not BaseDataset, raise an error
        if not issubclass(other_dataset.__class__, BaseDataset):
            raise ValueError("Other dataset must be a subclass of BaseDataset")
        
        self.ids.append(other_dataset.ids)
        self.roots.append(other_dataset.roots)
        
        # Combine images
        self.train_images = np.concatenate((self.train_images, other_dataset.train_images))
        self.test_images = np.concatenate((self.test_images, other_dataset.test_images))
        
        # Combine labels
        self.train_labels = np.concatenate((self.train_labels, other_dataset.train_labels))
        self.test_labels = np.concatenate((self.test_labels, other_dataset.test_labels))
        
        # Update crops and states
        self.crops = np.unique(np.concatenate((self.crops, other_dataset.crops)))
        self.states = np.unique(np.concatenate((self.states, other_dataset.states)))
        
        # Update the map
        self.train_map = self.generate_map('train')
        self.test_map = self.generate_map('test')
        
    def __str__(self):
        string = f"Dataset\n" + \
            f"IDs: {self.ids}\n" + \
            f"Root: {self.roots}\n" + \
            f"Train Images: {len(self.train_images)}\n" + \
            f"Test Images: {len(self.test_images)}\n" + \
            f"Crops: {self.crops}\n" + \
            f"States: {self.states}\n" + \
            f"Train Map: {self.train_map}\n" + \
            f"Test Map: {self.test_map}"
        return string

   
class CropCCMTDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def structure_data(self, test_split: float = None):
        
        train_images = []
        train_labels = []
        test_images = []
        test_labels = []
        
        # Get each directory name in the root directory
        crop_directories = os.listdir(self.roots[0])
        
        for crop in crop_directories:
            
            split_directory = os.listdir(os.path.join(self.roots[0], crop))
            
            for split in split_directory:
                
                state_directories = os.listdir(os.path.join(self.roots[0], crop, split))
                
                for state in state_directories:
                    
                    image_files = os.listdir(os.path.join(self.roots[0], crop, split, state))
                    
                    for image_file in image_files:
                        
                        if not image_file.lower().endswith('.jpg') and not image_file.lower().endswith('.jpeg'):
                            logger.debug(f'File {image_file} is not a jpg file, skipping')
                            continue
                        
                        image_path = os.path.join(self.roots[0], crop, split, state, image_file)
                        
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
        
        directories = os.listdir(self.roots[0])
        
        image_paths = []
        labels = []
        
        for directory in directories:
            label_split = directory.split('___')
            crop = label_split[0].title()
            state = label_split[1].title()
            
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
            
            
            for image in os.listdir(os.path.join(self.roots[0], directory)):
                if not image.lower().endswith('.jpg') and not image.lower().endswith('.jpeg'):
                    logger.debug(f'File {image} is not a jpg file, skipping')
                    continue
                
                image_path = os.path.join(self.roots[0], directory, image)
                image_paths.append(image_path)
                labels.append((crop, state))

        image_paths = np.array(image_paths, dtype=object)
        labels = np.array(labels, dtype=object)
        
        # Randomly split image paths and labels into train and test splits
        num_samples = len(image_paths)
        
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        
        split_idx = int(num_samples * (1 - test_split))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        train_image_paths = image_paths[train_indices]
        train_labels = labels[train_indices]
        test_image_paths = image_paths[test_indices]
        test_labels = labels[test_indices]
        
        return train_image_paths, train_labels, test_image_paths, test_labels
        
        
        

