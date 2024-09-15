from app.pipeline_helper.datasetadapter import CropCCMTDataset, PlantVillageDataset
from app.pipeline_helper.numpyarraymanager import NumpyArrayManager
import torch
from PIL import Image
import os
import json
from app.pipeline_helper.datasetadapter import Structure
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
        count = 0
        for dataset_config in config['datasets']:
            if not dataset_config['active']:
                continue
            dataset = self._load_dataset(dataset_config)
            count += 1
            if not self.dataset:
                self.dataset = dataset
            else:
                self.dataset.combine(dataset)
        
        # Obtain the loaded datasets structure
        new_structure = self.dataset.extract_structure()
        plural = 'Datasets have' if count > 1 else 'Dataset has'
        logger.info(f"{count} {plural} been loaded with structure - {new_structure}")
        
        # Check if an existing structure has been saved and update
        existing_structure = None
        if os.path.exists(os.path.join(self.output_directory, 'structure', 'structure.json')):
            existing_structure = self._load_existing_structure()
            logger.info(f"Existing saved structure found from past use - {existing_structure}")
            # Update the structure with new structure unique elements
        if existing_structure:
            existing_structure = self._update_structure(existing_structure, new_structure)

         
        # Save the updated structure
        self.structure = existing_structure if existing_structure else new_structure
        self.structure.save_structure(os.path.join(self.output_directory, 'structure'))

    
    def _update_structure(self, structure, new_structure):
        if not structure.equal(new_structure):
            logger.info(f"Current structure doesn't match new structure: {structure} -VS- {new_structure}")
            structure = structure.merge(new_structure)
            logger.info(f"Appended new structures unique values to existing structure to merge into: {structure}")
        else:
            logger.info("Structure has not changed")
        return structure
   
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
        crop_label = np.where(self.crops == crop)[0][0]
        state_label = np.where(self.states == state)[0][0]

        return img, crop_label, state_label
    
    def __len__(self) -> int:
        """ Method to return the length of the dataset

        Returns:
            int: The length of the dataset
        """
        return len(self.images)