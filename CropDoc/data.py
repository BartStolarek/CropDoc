import os
import shutil
import torch
#from loguru import logger
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose
#from app.config import AppConfig
#from app.utility.path import directory_exists
#from torch.utils.data import ConcatDataset
from typing import Union
import numpy as np
#from app.utility.path import find_directory
import matplotlib.pyplot as plt  # Plotting library
from typing import Dict, List
from PIL import Image
import re
from torchvision.transforms import (
    Compose,
    RandomResizedCrop,
    RandomHorizontalFlip,
    ColorJitter,
    RandomRotation,
    RandomAffine,
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
    Grayscale,
    RandomGrayscale,
    RandomPerspective,
    RandomVerticalFlip,
    FiveCrop,
    TenCrop,
    Lambda,
    Pad,
    RandomCrop,
    RandomErasing,
    GaussianBlur,
    LinearTransformation,
    Resize,
    CenterCrop,
    RandomRotation,
    RandomAffine,
    ColorJitter,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomResizedCrop,
    ToTensor,
    Normalize,
)
import torchvision.transforms as transforms


class SampleList(list):
    def __init__(self, samples: List[Dict] = []):
        super().__init__(samples)
        self.data_map = self._create_data_map(samples)
    
    def _create_data_map(self, samples: List[Dict]) -> Dict[str, Dict[str, int]]:
        #logger.debug(f"Creating data map")
        crop_map_dict = {}
        state_map_dict = {}
        for sample in samples:
            if sample['crop_label'] not in crop_map_dict:
                crop_map_dict[sample['crop_label']] = 1  # Initialize to 1 for the first occurrence
            else:
                crop_map_dict[sample['crop_label']] += 1
            
            if sample['state_label'] not in state_map_dict:
                state_map_dict[sample['state_label']] = 1  # Initialize to 1 for the first occurrence
            else:
                state_map_dict[sample['state_label']] += 1
        
        #logger.info(f"Created data map {crop_map_dict}, {state_map_dict}")
        return {'crops': crop_map_dict, 'states': state_map_dict}

  
class DatasetManager():
    """
    Manages a dataset located at a specified root path, providing access to samples,
    unique crop and state labels, and transformed samples for training and testing.
    
    The SampleList attributes have 'data_map' attribute that contains the number of samples for each crop and state.
    To access simply do: dataset.samples.data_map or dataset.train_samples.data_map

    Attributes:
        root_path (str): The root path of the dataset.
        transform (dict[str, Compose]): Transformation pipeline for dataset samples.
        samples (SampleList): List of all samples in the dataset.
        unique_crops (List[str]): List of unique crop labels in the dataset.
        unique_states (List[str]): List of unique state labels in the dataset.
        train_samples (SampleList): List of samples for training.
        test_samples (SampleList): List of samples for testing/validation.
    """
    def __init__(self, root_path: str, transform: dict[str, Compose]):
        self.root_path = root_path
        self.transform = transform
        self.samples = self._get_samples(root_path) # list of samples {img_path, split, crop_label, state_label, idx, crop_idx, state_idx}
        self.unique_crops = self._get_unique_crops(self.samples)
        self.unique_states = self._get_unique_states(self.samples)
        self.samples = self._add_idx_to_samples(self.samples)
        self.train_samples = SampleList(self._train_samples())
        self.test_samples = SampleList(self._test_samples())
        self.samples = self._add_idx_to_samples(self.samples)
        self.samples = self._add_crop_idx_to_samples(self.samples)
        self.samples = self._add_state_idx_to_samples(self.samples)
    
    def __str__(self):
        string = f"DatasetManager\n" + \
                    f"Root Path: {self.root_path}\n" + \
                    f"Transformers: {self.transform['length']}\n" + \
                    f"Unique Crops: {self.unique_crops}\n" + \
                    f"Unique States: {self.unique_states}\n" + \
                    f"Samples: {len(self.samples)}\n" + \
                    f"Train Samples: {len(self.train_samples)}\n" + \
                    f"Test Samples: {len(self.test_samples)}\n" + \
                    f"Data Map: {self.samples.data_map}"
        return string
    
    def _get_samples(self, root_path: str):
        #logger.debug(f"Getting samples from {root_path}")
        samples = []
        for root, directory, files in os.walk(root_path):
            for file in files:
                
                # Get the file name
                file_name = os.path.basename(file)
                
                # Convert file to sample format
                image_dict = self._convert_file_to_sample(root, file_name)
                
                # Add sample to samples list
                samples.append(image_dict)
        #logger.info(f"Obtained {len(samples)} samples")
        samples = SampleList(samples)
        return samples
    
    def _convert_file_to_sample(self, root: str, file_name: str):

        # Remove the digits from the start of the file name
        class_name = re.sub(r'^\d+', '', file_name)
             
        # Remove any extension type
        class_name = os.path.splitext(class_name)[0]

        # Get each label
        try:
            # Split the file name up by underscores
            labels = class_name.split('_')
            crop_label = labels[0]
            split = labels[1]
            state_label = labels[2]
        except Exception as e:
            #logger.error(f"Error splitting file name {labels}: {e}")
            return None
        
        # Keep healthy state labels separate between crops
        if state_label == 'healthy':
            state_label = crop_label + ' healthy'
            
        if split == 'valid':
            split = 'test'
        
        return {
            'img_path': os.path.join(root, file_name),
            'split': split,
            'crop_label': crop_label,
            'state_label': state_label
        }
    
    def _get_unique_crops(self, samples: SampleList) -> List[str]:
        return list(set([sample['crop_label'] for sample in samples]))
    
    def _get_unique_states(self, samples: SampleList) -> List[str]:
        return list(set([sample['state_label'] for sample in samples]))
       
    def __len__(self):
        return len(self.samples)
    
    def _add_idx_to_samples(self, samples: SampleList) -> SampleList:
        for idx, sample in enumerate(samples):
            sample['idx'] = idx
        return samples
    
    def _add_crop_idx_to_samples(self, samples: SampleList) -> SampleList:
        for idx, sample in enumerate(samples):
            sample['crop_idx'] = self.unique_crops.index(sample['crop_label'])
        return samples
    
    def _add_state_idx_to_samples(self, samples: SampleList) -> SampleList:
        for idx, sample in enumerate(samples):
            sample['state_idx'] = self.unique_states.index(sample['state_label'])
        return samples

    def __getitem__(self, idx):
        
        sample = self.samples[idx]
        img_path = sample['img_path']
        split = sample['split']
        crop_label = sample['crop_label']
        state_label = sample['state_label']
        
        image = plt.imread(img_path).convert('RGB')
        if split in self.transform.keys():
            image = self.transform[split](image)
        
        print("here")
        # Convert labels to appropriate tensor types
        crop_label = torch.tensor(self.unique_crops.index(sample['crop_label']), dtype=torch.long)
        state_label = torch.tensor(self.unique_states.index(sample['state_label']), dtype=torch.long)
        
        return image, crop_label, state_label
    
    def _train_samples(self) -> List[Dict]:
        return [sample for sample in self.samples if sample['split'] == 'train']
    
    def _test_samples(self) -> List[Dict]:
        return [sample for sample in self.samples if sample['split'] == 'test']

    
    def load_image_from_path(self, img_path: List[str], split: str):
        """
        Loads images from a list of image paths and applies transformations.

        Args:
            img_paths (List[str]): List of image paths to load.
            split (str): Indicates the split type ('train', 'val', 'test').

        Returns:
            List[torch.Tensor]: List of transformed images as PyTorch tensors.
        """
        image = Image.open(img_path).convert('RGB')
        if split in self.transform.keys():
            transformer = self.transform[split]
            image = transformer(image)
        
        return image
            
class TransformerManager():
    def __init__(self, transforms: dict[str, list[dict]]):
        self.transformers = self._get_transformers(transforms)
        self.transformers['length'] = len(self)
        
    def __len__(self):
        return self._calculate_total_transforms()
        
    def _get_transformers(self, transforms: dict[str, list[dict]]) -> dict[str, Compose]:
        #logger.debug(f"Getting transformers")
        if not transforms:
            #logger.warning(f"No transformers provided")
            return None
        
        if not isinstance(transforms, dict):
            #logger.error(f"Transformers must be a dictionary")
            return None
        
        if not set(['train', 'val', 'test']).issubset(transforms.keys()):
            #logger.error("Transformers must contain keys 'train', 'val', and 'test'")
            return None
        
        # Get train transformers
        train_transformers = self._build_transformers(transforms['train'])
        
        # Get val transformers
        val_transformers = self._build_transformers(transforms['val'])
        
        # Get test transformers
        test_transformers = self._build_transformers(transforms['test'])
        
        transformer_dict = {'train': train_transformers, 'val': val_transformers, 'test': test_transformers}
        
        #logger.info(f"Obtained transformers dict {transformer_dict}")
        
        return transformer_dict
    
    def _build_transformers(self, transformer_list: List[dict]) -> Compose:
        #logger.debug(f"Building transformers")
        if not transformer_list:
            #logger.warning(f"No transformers provided")
            return None
        
        if not isinstance(transformer_list, list):
            #logger.error(f"Transformers must be a list")
            return None
        
        transforms_list = []
        for transform_dict in transformer_list:
            try:
                transform = self._parse_transform(transform_dict)
                if transform:
                    transforms_list.append(transform)
            except Exception as e:
                None
                #logger.error(f"Error parsing transform '{transform_dict}': {e}")
        
        #logger.info(f"Built transformers list {transforms_list}")
        return Compose(transforms_list)
    
    def _parse_transform(self, transform_dict: dict):
        #logger.debug(f"Parsing transform {transform_dict}")
        transform_name = transform_dict['type']
        
        # Remove type and its value from dict
        transform_dict.pop('type')
        
        transform_fn = getattr(transforms, transform_name)
        #logger.info(f"Converted transformer dict to {transform_fn}")
        return transform_fn(**transform_dict)
        
    def _calculate_total_transforms(self) -> int:
        total_transforms = 0
        for key, value in self.transformers.items():
            if isinstance(value, Compose):
                total_transforms += len(value.transforms)
        return total_transforms