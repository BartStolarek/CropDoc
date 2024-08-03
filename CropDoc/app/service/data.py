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
from app.utility.path import find_directory
import matplotlib.pyplot as plt  # Plotting library
from typing import Dict, List


class SampleList(list):
    def __init__(self, samples: List[Dict]):
        super().__init__(samples)
        self.data_map = self._create_data_map(samples)

    def _create_data_map(self, samples: List[Dict]) -> Dict[str, Dict[str, int]]:
        unique_crops = set([sample['crop_label'] for sample in samples])
        unique_states = set([sample['state_label'] for sample in samples])
        
        crop_counts = {crop: 0 for crop in unique_crops}
        state_counts = {state: 0 for state in unique_states}
        
        for sample in samples:
            crop_counts[sample['crop_label']] += 1
            state_counts[sample['state_label']] += 1
        
        return {'crops': crop_counts, 'states': state_counts}
    
    
class DatasetManager():
    def __init__(self, root_path: str, transform: dict[str, Compose]):
        self.root_path = root_path
        self.transform = transform
        self.samples = self._get_samples(root_path)
        self.unique_crops = self._get_unique_crops(self.samples)
        self.unique_states = self._get_unique_states(self.samples)
        self.samples = self._add_idx_to_samples(self.samples)
        self.data_map = self._create_data_map(self.samples)
        self.train_samples = SampleList(self._train_samples())
        self.test_samples = SampleList(self._test_samples())
    
    def _get_samples(self, root_path: str):
        samples = SampleList()
        for root, directory, files in os.walk(root_path):
            for file in files:
                
                # Get the file name
                file_name = os.path.basename(file)
                
                # Convert file to sample format
                image_dict = self._convert_file_to_sample(root, file_name)
                
                # Add sample to samples list
                samples.append(image_dict)
                
        return samples
    
    def _convert_file_to_sample(self, root: str, file_name: str):
        # Remove the digits from the start of the file name
        for idx, letter in enumerate(file_name):
            if not letter.isdigit():
                file_name = file_name[idx:]
                
        # Remove the extension .jpg from the file name
        file_name = file_name.replace('.jpg', '')
        
        # Split the file name up by underscores
        labels = file_name.split('_')
        
        # Get each label
        crop_label = labels[0]
        split = labels[1]
        state_label = labels[2]
        
        # Keep healthy state labels separate between crops
        if state_label == 'healthy':
            state_label = crop_label + '_healthy'
        
        return {
            'img_path': os.path.join(root, file_name),
            'split': split,
            'crop_label': crop_label,
            'state_label': state_label
        }
    
    def _get_unique_crops(self, samples):
        return set([sample['crop_label'] for sample in samples])
    
    def _get_unique_states(self, samples):
        return set([sample['state_label'] for sample in samples])
        
    def _add_idx_to_samples(self, samples):
        for sample in samples:
            sample['crop_idx'] = self.unique_crops.index(sample['crop_label'])
            sample['state_idx'] = self.unique_states.index(sample['state_label'])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = plt.imread(sample['img_path'])
        if self.transform:
            img = self.transform[sample['split']](img)
        return img, sample['crop_idx'], sample['state_idx']

    def _train_samples(self) -> List[Dict]:
        return [sample for sample in self.samples if sample['split'] == 'train']
    
    def _test_samples(self) -> List[Dict]:
        return [sample for sample in self.samples if sample['split'] == 'test']
    
    def _create_data_map(self, split: str = None) -> Dict[str, Dict[str, int]]:
        if split:
            samples = [sample for sample in self.samples if sample['split'] == split]
        else:
            samples = self.samples
        
        crop_counts = {crop: 0 for crop in self.unique_crops}
        state_counts = {state: 0 for state in self.unique_states}
        
        for sample in samples:
            crop_counts[sample['crop_label']] += 1
            state_counts[sample['state_label']] += 1
        
        return {'crops': crop_counts, 'states': state_counts}
    
    
        
        

class ResNet50V2Subset():
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


