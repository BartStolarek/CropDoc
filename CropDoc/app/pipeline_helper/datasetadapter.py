import os
import re
from abc import ABC, abstractmethod
from app.pipeline_helper.numpyarraymanager import NumpyArrayManager

import numpy as np
import torch
from loguru import logger
from PIL import Image
import json

PIL_SUPPORTED_EXTENSIONS = [
    'jpg',
    'jpeg',
    'png',
    'gif',
    'bmp',
    'tiff',
    'tif',
    'webp',
    'ico',
    'ppm',  # Portable Pixmap
    'pgm',  # Portable Graymap
    'pbm',  # Portable Bitmap
    'pcx',  # PCX
    'tga',  # Truevision TGA
    'eps',  # Encapsulated Postscript (requires Ghostscript)
]


class Structure():
    def __init__(self, train_images=None, train_labels=None, test_images=None, test_labels=None, crops=None, states=None):
        self.train_images = np.array(train_images) if not isinstance(train_images, np.ndarray) else train_images
        self.train_labels = np.array(train_labels) if not isinstance(train_labels, np.ndarray) else train_labels
        self.test_images = np.array(test_images) if not isinstance(test_images, np.ndarray) else test_images
        self.test_labels = np.array(test_labels) if not isinstance(test_labels, np.ndarray) else test_labels
        self.crops = np.array(crops) if not isinstance(crops, np.ndarray) else crops
        self.states = np.array(states) if not isinstance(states, np.ndarray) else states
        
    def save_structure(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        save_dict = self.convert_to_dict()
        
        with open(os.path.join(directory, 'structure.json'), 'w') as f:
            json.dump(save_dict, f)
        logger.info(f"Structure saved at {os.path.join(directory, 'structure.json')}")
    
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
    
    def equal(self, other_structure) -> False:
        np_array_manager = NumpyArrayManager()
        
        for this_key, this_value in self.__dict__.items():
            other_value = getattr(other_structure, this_key)
            
            this_value_unique = np.unique(this_value)
            other_value_unique = np.unique(other_value)
            
            if len(this_value_unique) != len(other_value_unique):
                logger.info(f"Current structure {this_key} has different length of unique values ({len(this_value_unique)}) than other structure ({len(other_value_unique)})")
                return False
            
            this_value_unique = np.sort(this_value_unique)
            other_value_unique = np.sort(other_value_unique)
            
            if not np.array_equal(this_value_unique, other_value_unique):
                logger.info(f"Current structure {this_key} has different unique values than other structure")
                difference = np_array_manager.get_difference(this_value_unique, other_value_unique)
                # Print first 10 values of difference
                logger.info(f"First 10 differences: {difference[:10]}")
                logger.info(f"Other structure has {len(difference)} unique values")
                return False
            
        return True
        
    def merge(self, other_structure):
        np_array_manager = NumpyArrayManager()
        
        new_train_images, train_appended_indices = np_array_manager.append_missing_unique_elements(self.train_images, other_structure.train_images)
        logger.info(f"Found {len(train_appended_indices)} new train images to append")
        new_test_images, test_appended_indices = np_array_manager.append_missing_unique_elements(self.test_images, other_structure.test_images)
        logger.info(f"Found {len(test_appended_indices)} new test images to append")
        
        new_train_labels = np.concatenate((self.train_labels, other_structure.train_labels[train_appended_indices]))
        new_test_labels = np.concatenate((self.test_labels, other_structure.test_labels[test_appended_indices]))
        
        new_crops, _ = np_array_manager.append_missing_unique_elements(self.crops, other_structure.crops)
        new_states, _ = np_array_manager.append_missing_unique_elements(self.states, other_structure.states)
        
        return Structure(
            train_images=new_train_images,
            train_labels=new_train_labels,
            test_images=new_test_images,
            test_labels=new_test_labels,
            crops=new_crops,
            states=new_states
        )

     
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
        logger.info(f"Found {len(self.crops)} unique crops and {len(self.states)} unique states")
        
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
        train_crops = np.sort(np.unique(self.train_labels[:, 0]))
        test_crops = np.sort(np.unique(self.test_labels[:, 0]))
        
        # Check if the train and test crops have exact same crops
        if not np.array_equal(train_crops, test_crops):
            raise ValueError(f"Train and test crops must be the same, there are different crops in the train and test splits: \nTrain Crops:\n{train_crops}, \nTest Crops:\n{test_crops}")
        

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
            raise ValueError(f"Train and test states must be the same, there are different states in the train and test splits: \nTrain states:\n{train_states}, \nTest states:\n{test_states}")
        

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
        
        train_objects = []
        
        test_objects = []
        
        # Get each directory name in the root directory
        crop_directories = sorted(os.listdir(self.roots[0]))
        
        for crop in crop_directories:
            
            split_directory = sorted(os.listdir(os.path.join(self.roots[0], crop)))
            
            for split in split_directory:
                
                state_directories = sorted(os.listdir(os.path.join(self.roots[0], crop, split)))
                
                for state in state_directories:
                    
                    image_files = sorted(os.listdir(os.path.join(self.roots[0], crop, split, state)))
                    
                    for image_file in image_files:
                        
                        # Obtain image file extension
                        extension = os.path.splitext(image_file)[1]
                        
                        if not extension.lower() not in PIL_SUPPORTED_EXTENSIONS:
                            logger.debug(f'File {image_file} is not a image file, skipping')
                            continue
                        
                        image_path = os.path.join(self.roots[0], crop, split, state, image_file)
                        
                        crop_label = crop.title()
                        
                        
                        
                        state_label = crop_label + '-Healthy' if 'healthy' in state.lower() else state.title()
                        
                        # Remove digits from state label
                        state_label = re.sub(r'\d+', '', state)
                        
                        if 'test' in split:
                            test_objects.append((image_path, (crop_label, state_label)))
                        elif 'train' in split:
                            train_objects.append((image_path, (crop_label, state_label)))

        
        # Sort by image_path
        train_objects = sorted(train_objects, key=lambda x: x[0])
        test_objects = sorted(test_objects, key=lambda x: x[0])
        
        
        # Unpack the sorted objects in to np arrays
        train_images = np.array([x[0] for x in train_objects])
        train_labels = np.array([x[1] for x in train_objects])
        test_images = np.array([x[0] for x in test_objects])
        test_labels = np.array([x[1] for x in test_objects])
        
        logger.info(f'Finished structuring CCMT Dataset data, found {len(train_images)} train images and {len(test_images)} test images')
        
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
        
        directories = sorted(os.listdir(self.roots[0]))
        
        image_paths = []
        labels = []
        image_objects= []
        
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
            
            image_files = sorted(os.listdir(os.path.join(self.roots[0], directory)))
            for image in image_files:
                extension = os.path.splitext(image)[1]
                if not extension.lower() not in PIL_SUPPORTED_EXTENSIONS:
                    logger.debug(f'File {image} is not a image file, skipping')
                    continue
                
                image_path = os.path.join(self.roots[0], directory, image)
                image_objects.append((image_path, (crop, state)))

        # Sort the image objects by image path
        image_objects = sorted(image_objects, key=lambda x: x[0])

        # Unpack the sorted objects into numpy arrays
        image_paths = np.array([x[0] for x in image_objects])
        labels = np.array([x[1] for x in image_objects])
        
        # Randomly split image paths and labels into train and test splits
        num_samples = len(image_paths)
        
        # Set a fixed random seed
        np.random.seed(42)
        
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        
        split_idx = int(num_samples * (1 - test_split))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        train_image_paths = image_paths[train_indices]
        train_labels = labels[train_indices]
        test_image_paths = image_paths[test_indices]
        test_labels = labels[test_indices]
        
        logger.info(f'Finished structuring Plant Village Dataset data, found {len(train_image_paths)} train images and {len(test_image_paths)} test images')
        
        return train_image_paths, train_labels, test_image_paths, test_labels
        
        
        

