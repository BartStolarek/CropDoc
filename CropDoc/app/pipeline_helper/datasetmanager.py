from app.pipeline_helper.dataset import CropCCMTDataset, PlantVillageDataset
import torch
from PIL import Image
import os
import json
from app.pipeline_helper.dataset import Structure
from loguru import logger
import numpy as np
from collections import Counter

DATASET_CLASSES = {
    'CropCCMTDataset': CropCCMTDataset,
    'PlantVillageDataset': PlantVillageDataset
}


class DatasetManager():
    
    def __init__(self, config, output_directory):
        self.output_directory = output_directory
        self.new_head_required = False
        
        # Load the datasets
        self.dataset = None
        for dataset_config in config['datasets']:
            if not dataset_config['active']:
                continue
            dataset = self._load_dataset(dataset_config)
            if not self.dataset:
                self.dataset = dataset
            else:
                self.dataset.combine(dataset)
        
        # Obtain the loaded datasets structure
        new_structure = self.dataset.extract_structure()
        logger.info(f"Loaded dataset with structure - {new_structure}")
        
        # Check if an existing structure has been saved and update
        existing_structure = None
        if os.path.exists(os.path.join(self.output_directory, 'structure', 'structure.json')):
            existing_structure = self._load_existing_structure()
            logger.info(f"Existing structure found - {existing_structure}")
            # Update the structure with new structure unique elements
        if existing_structure:
            existing_structure = self._update_structure(existing_structure, new_structure)
            logger.info(f"Updated structure - {existing_structure}")
         
        # Save the updated structure
        self.structure = existing_structure if existing_structure else new_structure
        self.structure.save_structure(os.path.join(self.output_directory, 'structure'))

    
    def _arrays_have_same_elements(self, a, b):
        return np.all(np.isin(a, b)) and np.all(np.isin(b, a))
    
    def _update_structure(self, existing_structure, new_structure):
        def update_arrays(existing_arr, new_arr, existing_labels=None, new_labels=None):
            existing_set = set(map(tuple, existing_arr))
            new_items = [item for item in new_arr if tuple(item) not in existing_set]
            
            if len(new_items) > 0:
                new_items = np.array(new_items)
                updated_arr = np.concatenate((existing_arr, new_items))
                
                if existing_labels is not None and new_labels is not None:
                    new_indices = [i for i, item in enumerate(new_arr) if tuple(item) not in existing_set]
                    updated_labels = np.concatenate((existing_labels, new_labels[new_indices]))
                    return updated_arr, updated_labels, len(new_items)
                
                return updated_arr, len(new_items)
            
            return existing_arr, 0

        # Update train images and labels
        existing_structure.train_images, existing_structure.train_labels, train_count = update_arrays(
            existing_structure.train_images, new_structure.train_images,
            existing_structure.train_labels, new_structure.train_labels
        )
        logger.info(f"Appended {train_count} new train images")

        # Update test images and labels
        existing_structure.test_images, existing_structure.test_labels, test_count = update_arrays(
            existing_structure.test_images, new_structure.test_images,
            existing_structure.test_labels, new_structure.test_labels
        )
        logger.info(f"Appended {test_count} new test images")

        # Update crops
        existing_structure.crops, crops_count = update_arrays(existing_structure.crops, new_structure.crops)
        logger.info(f"Appended {crops_count} new crops")

        # Update states
        existing_structure.states, states_count = update_arrays(existing_structure.states, new_structure.states)
        logger.info(f"Appended {states_count} new states")

        return existing_structure
         
    def _load_existing_structure(self):
        existing_structure = Structure()
        existing_structure.load_structure(os.path.join(self.output_directory, 'structure'))
        return existing_structure
                      
    def _load_dataset(self, dataset_config, existing_structure=None):
        name = dataset_config['name']
        dclass = dataset_config['class']
        root = dataset_config['root']
        
        if dclass not in DATASET_CLASSES:
            raise ValueError(f"Unknown dataset class {dclass}")
        
        dataset_class = DATASET_CLASSES[dclass]
        kwargs = {}
        
        if existing_structure:
            kwargs['existing_structure'] = existing_structure
        
        if dclass == 'PlantVillageDataset':
            kwargs['test_split'] = dataset_config['test_split']
        
        return dataset_class(root=root, name=name, **kwargs)
        
            
    def get_train_dataset(self):
        images = self.dataset.train_images
        labels = self.dataset.train_labels
        crops = self.dataset.crops
        states = self.dataset.states
        return Dataset(images, labels, crops, states)
    
    def get_test_dataset(self):
        images = self.dataset.test_images
        labels = self.dataset.test_labels
        crops = self.dataset.crops
        states = self.dataset.states
        return Dataset(images, labels, crops, states)
            
    
class Dataset():
    
    def __init__(self, images, labels, crops, states):
        self.images = images
        self.labels = labels
        self.crops = crops
        self.states = states
        
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