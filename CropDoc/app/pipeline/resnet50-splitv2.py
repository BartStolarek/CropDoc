import os
import re
import time
import warnings
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from loguru import logger
from PIL import Image
from tqdm import tqdm

from app.pipeline_helper.transformer import TransformerManager
from app.pipeline_helper.dataset import CropCCMTDataset


class MultiHeadResNetModel(torch.nn.Module):
    """A multi-head ResNet model for the CropCCMT dataset

    Args:
        torch (torch.nn.Module): The PyTorch module
    """
    def __init__(self, num_classes_crop, num_classes_state):
        """ Initialise a multi-head ResNet50 model with;
        - A ResNet50 backbone and pre-trained weights
        - A crop head
        - A state head
        
        Also move the model to the GPU if available

        Args:
            num_classes_crop (int): The number of unique classes for the crop head
            num_classes_state (int): The number of unique classes for the state head
        """
        super(MultiHeadResNetModel, self).__init__()
        
        # Check if GPU is available
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the ResNet50 model with pre-trained weights
        self.resnet = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.DEFAULT)  # TODO: Add to report why we used ResNet50 default weights and benefits
        
        # Modify the model to remove the final fully connected layer
        num_ftres = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Identity()
        
        # Add the crop and state heads
        self.crop_fc = torch.nn.Linear(num_ftres, num_classes_crop) 
        self.state_fc = torch.nn.Linear(num_ftres, num_classes_state)  # TODO: Add to report that we replaced the final layer and why, and how it is multihead

        # Move all parts of the model to the same device
        self.resnet = self.resnet.to(self.device)
        self.crop_fc = self.crop_fc.to(self.device)
        self.state_fc = self.state_fc.to(self.device)

        # Wrap only the resnet part in DataParallel
        self.resnet = torch.nn.DataParallel(self.resnet)

    def forward(self, x):
        """ Forward pass through the model, and return the tensors for the crop and state heads as a tuple

        Args:
            x (torch.Tensor): The input tensor, where x.shape is torch.Size(<batch_size>, <num_features>)

        Returns:
            tuple: A tuple containing the crop and state tensors
        """
        x = x.to(self.device)  # Move input to GPU if available
        
        # Forward pass through the ResNet backbone
        x = self.resnet(x)
        
        # Forward pass through the crop and state heads
        crop_out = self.crop_fc(x)
        state_out = self.state_fc(x)
        
        # Return the crop and state tensors
        return crop_out, state_out  # TODO: Add to report that the forward pass will return the crop and state tensors


class Pipeline():
    
    def __init__(self, config: dict):
        
        # Check if the config is valid
        self._validate_config(config)
        
        # Split out the config for easier interpretation
        self.model_config = config['model']
        self.pipeline_config = config['pipeline']
        
        # Get the dataset root directory
        self.dataset_root = self.pipeline_config['dataset_dir']
        self._validate_dataset_root(self.dataset_root)
        
    def train_model(self):
        
        logger.debug("Starting model training")
        
        # Load the transformer manager to handle loading transformers from config
        self.transformer_manager = TransformerManager(
            self.model_config['data']['transformers']  # TODO: Add to report that we are using transformers to augment the training data
        )
        
        # Load the train datasets
        self.train_data = CropCCMTDataset(
            dataset_path=self.dataset_root,
            transformers=self.transformer_manager.transformers['train'],
            split='train'
        )
        
        # If in development dataset needs to be reduced for faster training
        development_reduction = self.model_config['data']['reduce']
        if development_reduction < 1:
            self.train_data.equal_reduce(development_reduction)
            
        # Define the model
        self.model = MultiHeadResNetModel(
            num_classes_crop=self.train_data.get_unique_crop_count(),  # TODO: Add to report the count of unique classes for crop and state
            num_classes_state=self.train_data.get_unique_state_count()
        )
        
        # Define the loss functions for both heads
        self.crop_criterion = torch.nn.CrossEntropyLoss()  # TODO: Add to report that we are using CrossEntropyLoss for the loss functions for both heads
        self.state_criterion = torch.nn.CrossEntropyLoss()
        
        # Define the optimiser
        self.optimiser = torch.optim.Adam(  # TODO: Add to the report that we are using Adam optimiser, and the learning rate it starts at
            self.model.parameters(),
            lr=self.model_config['training']['learning_rate']
        )
        
        # Define the learning rate scheduler, which will reduce the learning rate when the model stops improving so that it can find a better minima
        active_scheduler = self.model_config['training']['lr_scheduler']['active']  # Obtain the active scheduler set in the config
        scheduler_config = self.model_config['training']['lr_scheduler'][active_scheduler]  # Obtain the configuration for the active scheduler
        scheduler_object = getattr(torch.optim.lr_scheduler, active_scheduler)  # TODO: Add to the report the scheduler we are using (from config file)
        self.scheduler = scheduler_object(
            self.optimizer,
            **scheduler_config
        )
        
        logger.info('Model Initialisation Complete')
        
        logger.debug('Starting Training Loop')
        
        # Define the number of epochs to train for
        epochs = self.model_config['training']['epochs']
        
        # Define the best validation loss
        self.best_val_loss = np.inf  # Set the best validation loss to infinity so that the first validation loss will always be better
        
        # Start training loop
        for i in tqdm(epochs, desc="Epoch", leave=True):  # TODO: Add to the report the number of epochs we trained for
            
            # Train the model for one epoch
            epoch_metrics = self._train_one_epoch(idx=i)
            
            # Update the learning rate
            val_loss = epoch_metrics['val']['loss']['combined']
            self.scheduler.step(val_loss)
            
            # Save the model if it has improved
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(output_dir=self.pipeline_config['output_dir'], epoch=i)
            
            
            
            
            
    
    def _train_one_epoch(self, idx: int) -> dict:
        pass
        
        
        
    
    def save_model(self):
        pass
    
    def save_index_map(self):
        """ Save the index map, which is a dictionary of the class names and their corresponding indices
        """
        pass
    
    def save_progression(self):
        """ Save the progression of the model training which is the change in loss and accuracy over epochs
        """
        pass
    
    def save_metrics(self):
        """ Save the following metrics:
        - Accuracy
        - Precision
        - Recall
        - F1 Score
        - Confusion Matrix
        """
        pass
    
    def save_graphs(self):
        """ Save the following graphs:
        - ROC Curve
        - AUC-ROC Curve
        - Training / Validation loss over epochs
        - Training / Validation accuracy over epochs
        - Confusion Matrix heatmap
        """
        pass
    
    def package(self):
        """ Package the pipeline into a object that can be returned via the API including;
        - Index Map
        - Progression
        - Performance Metrics
        - Evaluation Graphs
        """
        pass
    
    def _validate_config(self, config: dict):
        """ Validate the configuration dictionary to ensure that it contains all the necessary keys and values
        
        Args:
            config (dict): A dictionary containing the configuration for the training pipeline
        
        Raises:
            ValueError: If the configuration dictionary is invalid
        """
        pass
    
    def _validate_dataset_root(self, dataset_root: str):
        """ Validate the dataset root directory to ensure that it exists
        
        Args:
            dataset_root (str): The root directory of the dataset
        
        Raises:
            ValueError: If the dataset root directory does not exist or the structure is incorrect
        """
        pass


def train(config):
    """ A method/function to run the training pipeline

    Args:
        config (dict): A dictionary containing the configuration for the training pipeline

    Returns:
        _type_: _description_
    """
    logger.debug("Running the training pipeline")
    
    # Initialise the Pipeline
    pipeline = Pipeline(config)
    
    # Train the modeol
    pipeline.train_model()
    
    # Save the model
    pipeline.save_model()
    
    # Save the index map
    pipeline.save_index_map()
    
    # Save the progression
    pipeline.save_progression()
    
    # Save the performance metrics
    pipeline.save_metrics()
    
    # Save the evaluation graphs
    pipeline.save_graphs()
    
    pipeline_package = pipeline.package()
    
    logger.info("Training pipeline completed")
    
    return pipeline_package
