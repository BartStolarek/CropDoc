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

from pprint import pprint


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
        
        # Split out the config for easier interpretation, and get some pipeline metadata
        self.pipeline_config = config
        self.pipeline_name = self.pipeline_config['name']
        self.pipeline_version = self.pipeline_config['version']
        self.pipeline_output_dir = os.path.join(
            self.pipeline_config['output_dir'],
            f"{self.pipeline_name}-{self.pipeline_version}"
        )
        os.makedirs(self.pipeline_output_dir, exist_ok=True)
        
        # Get the dataset root directory
        self.dataset_root = self.pipeline_config['dataset_dir']
        self._validate_dataset_root(self.dataset_root)

    def train_model(self):
        """ Starts training the model and determining the best performing model. 
        The model will be trained for the number of epochs specified in the config file
        """
        logger.debug("Starting training the model")
        
        # Check if pre-existing pipeline exists
        potential_path = os.path.join(self.pipeline_output_dir, 'final', f'{self.pipeline_name}-{self.pipeline_version}.pth')
        if os.path.exists(potential_path):
            logger.info(f"Found a pre-existing pipeline for this config at {self.pipeline_output_dir}/final/")
            pipeline_exists = True
            self.pipeline_path = potential_path
        else:
            pipeline_exists = False
            logger.info("No pre-existing pipeline found, you can move a checkpoint file to the final directory, make sure to remove the 'epoch-#' post-fix")
        
        # Load the transformer manager to handle loading transformers from config
        self.transformer_manager = TransformerManager(
            self.pipeline_config['data']['transformers']  # TODO: Add to report that we are using transformers to augment the training data
        )
        
        # Load the train datasets
        dataset_kwargs = {
            'dataset_path': self.dataset_root,
            'transformers': self.transformer_manager.transformers['train'],
            'split': 'train'
        }
        if pipeline_exists:
            # Load the crop and state index map from the saved file
            crop_index_map = torch.load(self.pipeline_path, weights_only=True)['crop_index_map']
            state_index_map = torch.load(self.pipeline_path, weights_only=True)['state_index_map']
            
            # Give it to the dataset class to keep mapping the same
            dataset_kwargs['crop_index_map'] = crop_index_map
            dataset_kwargs['state_index_map'] = state_index_map
        
        # Initialise the training dataset
        self.train_data = CropCCMTDataset(
            **dataset_kwargs
        )
        
        # Resave the crop and state index maps
        self.crop_index_map = self.train_data.crop_index_map
        self.state_index_map = self.train_data.state_index_map
        
        logger.info(f"Loaded training dataset {self.train_data}")
        
        # If in development dataset needs to be reduced for faster training
        development_reduction = self.pipeline_config['data']['reduce']
        if development_reduction < 1:
            self.train_data.equal_reduce(development_reduction)
            
        # Define the model
        self.model = MultiHeadResNetModel(
            num_classes_crop=self.train_data.get_unique_crop_count(),  # TODO: Add to report the count of unique classes for crop and state
            num_classes_state=self.train_data.get_unique_state_count()
        )
        if pipeline_exists:
            self.model.load_state_dict(torch.load(self.pipeline_path, weights_only=True)['model_state_dict'])
        
        # Define the loss functions for both heads
        self.crop_criterion = torch.nn.CrossEntropyLoss()  # TODO: Add to report that we are using CrossEntropyLoss for the loss functions for both heads
        self.state_criterion = torch.nn.CrossEntropyLoss()
        
        # Define the optimiser
        self.optimiser = torch.optim.Adam(  # TODO: Add to the report that we are using Adam optimiser, and the learning rate it starts at
            self.model.parameters(),
            lr=self.pipeline_config['training']['learning_rate']
        )
        if pipeline_exists:
            self.optimiser.load_state_dict(torch.load(self.pipeline_path, weights_only=True)['optimiser_state_dict'])
        
        # Define the learning rate scheduler, which will reduce the learning rate when the model stops improving so that it can find a better minima
        active_scheduler = self.pipeline_config['training']['lr_scheduler']['active']  # Obtain the active scheduler set in the config
        scheduler_config = self.pipeline_config['training']['lr_scheduler'][active_scheduler]  # Obtain the configuration for the active scheduler
        scheduler_object = getattr(torch.optim.lr_scheduler, active_scheduler)  # TODO: Add to the report the scheduler we are using (from config file)
        self.scheduler = scheduler_object(
            self.optimiser,
            **scheduler_config
        )
        if pipeline_exists:
            self.scheduler.load_state_dict(torch.load(self.pipeline_path, weights_only=True)['scheduler_state_dict'])
        
        logger.info('Pipeline Initialisation Complete')
        
        logger.info('\nTraining loop index:\n' +
                    "T: Train, V: Validation\n" +
                    "C: Crop, S: State\n" +
                    "L: Loss, A: Accuracy\n"
                    )
        
        # Define the best validation loss
        checkpoint_interval = self.pipeline_config['training']['checkpoint_interval']  # The minimum epochs to go buy before checking to save another checkpoint

        # Define the number of epochs to train for and set the epoch range
        self.epochs = self.pipeline_config['training']['epochs']
        if pipeline_exists:
            start = torch.load(self.pipeline_path, weights_only=True)['epochs'] + 1
            end = start + self.epochs
            self.epochs += torch.load(self.pipeline_path, weights_only=True)['epochs']
            logger.info(f'Starting training loop for epochs {start} to {end - 1} ({end - 1 - start}). Total pre-trained epochs: {start - 1})')
            end -= 1
        else:
            start = 1
            end = self.epochs + 1
            logger.info(f'Starting training loop for epochs {start} to {end - 1}')
        
        
        
        # Initialise epoch progress bar and training metrics list
        epochs_progress = tqdm(range(start, end), desc="Epoch", leave=True)
        
        if pipeline_exists:
            self.progression_metrics = torch.load(self.pipeline_path, weights_only=True)['progression_metrics']
            self.performance_metrics = torch.load(self.pipeline_path, weights_only=True)['performance_metrics']
            
        else:
            self.progression_metrics = []  # Progression metrics will be a list of epoch number and the epochs metrics
        
        # Start training loop
        if pipeline_exists:
            last_checkpoint_epoch = start - 1
            first_epoch = False
        else:
            first_epoch = True
        for i in epochs_progress:  # TODO: Add to the report the number of epochs we trained for
            
            # Train the model for one epoch
            epoch_metrics = self._train_one_epoch()
            
            # Update the learning rate
            val_loss = epoch_metrics['val']['loss']['combined']
            self.scheduler.step(val_loss)
            
            # Check if its first epoch, if so save a checkpoing
            if first_epoch:
                last_checkpoint_epoch = i
                self.performance_metrics = epoch_metrics
                self.performance_metrics['epoch'] = i
                self.save_pipeline(epoch=i)
                
            # Check if the current epoch is far enough from previous checkpoint and also past halfway for epoch
            elif (i - last_checkpoint_epoch) >= checkpoint_interval and i >= end / 4:
                # Check if current epoch has better validation loss than current best
                if self._is_better_metric(epoch_metrics['val']):
                    self.performance_metrics = epoch_metrics
                    self.performance_metrics['epoch'] = i
                    self.save_pipeline(epoch=i)
                    last_checkpoint_epoch = i
            
            # Update the progress bar with the metrics
            epochs_progress.set_postfix(
                self._format_metrics(
                    metrics=[
                        epoch_metrics['train'],
                        epoch_metrics['val']
                    ],
                    dataset_names=['train', 'val']
                )
            )
            
            # Append the metrics to the training metrics list
            self.progression_metrics.append([i, epoch_metrics])
            
        
        
            first_epoch = False
        
        logger.info('Training loop complete')
        
        # Remove batches from performance metrics
        if 'train' in self.performance_metrics.keys() and 'batches' in self.performance_metrics['train'].keys():
            self.performance_metrics['train'].pop('batches')
        if 'val' in self.performance_metrics.keys() and 'batches' in self.performance_metrics['val'].keys():
            self.performance_metrics['val'].pop('batches')
        if 'test' in self.performance_metrics.keys() and 'batches' in self.performance_metrics['test'].keys():
            self.performance_metrics['test'].pop('batches') 
            
        
            
        logger.info(f"Best performance metrics :\n {self.performance_metrics}")
        
        logger.info('Training pipeline complete')

    def save_pipeline(self, epoch: int = None):
        """ Save the pipeline, including the model, optimiser, scheduler, progression metrics, and performance metrics
        as a checkpoint or the final pipeline

        Args:
            epoch (int, optional): The epoch number to save the checkpoint. Defaults to None if saving the final pipeline.
        """
        
        # Create the save dictionary
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimiser_state_dict': self.optimiser.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'progression_metrics': self.progression_metrics,
            'performance_metrics': self.performance_metrics,
            'crop_index_map': self.crop_index_map,
            'state_index_map': self.state_index_map,
        }
        
        # Save the pipeline as a checkpoint or the final pipeline depending if a epoch was provided
        if epoch is not None:
            directory = os.path.join(self.pipeline_output_dir, 'checkpoints')
            file_name = f'{self.pipeline_name}-{self.pipeline_version}-epoch-{epoch}.pth'
            save_dict['epochs'] = epoch
        else:
            directory = os.path.join(self.pipeline_output_dir, 'final')
            file_name = f'{self.pipeline_name}-{self.pipeline_version}.pth'
            save_dict['epochs'] = self.epochs
            
        
        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)
        
        # Save the pipeline
        file_path = os.path.join(directory, file_name)
        torch.save(save_dict, file_path)
        
        if not epoch:
            logger.info(f"Saved the final pipeline to {file_path}")

    
    def save_confusion_matrix(self):
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
    
    def _load_model(self):
        torch
    
    def _train_one_epoch(self) -> dict:
        """ Train the model for one epoch, splitting the training data into a training and validation set

        Args:
            idx (int): The index of the epoch

        Returns:
            dict: A dictionary containing the performance metrics for this epoch's training and validation
        """
        
        epoch_metrics = {}
        
        # Split the train dataset into a training and validation set
        validation_split = self.pipeline_config['training']['validation_split']
        train_dataset, val_dataset = torch.utils.data.dataset.random_split(
            self.train_data,
            [
                1 - validation_split,
                validation_split
            ]
        )
        
        # Create the dataloaders for the training and validation sets that will load in to the model
        train_dataloader = self._get_dataloader(dataset=train_dataset, shuffle=True)
        val_dataloader = self._get_dataloader(dataset=val_dataset, shuffle=False)
        
        # Set the model to train mode
        self.model.train()
        
        # Feed the model the training data, and set feeder to train
        train_metrics = self._feed_model(train_dataloader, train=True)
        epoch_metrics['train'] = train_metrics
        
        # Set the model to evaluation/validation mode
        self.model.eval()
        
        # Turn off gradients for validation
        with torch.no_grad():
            
            # Feed the model the validation data
            val_metrics = self._feed_model(val_dataloader)
            epoch_metrics['val'] = val_metrics
        
        # If using GPU clear the cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return epoch_metrics
        
    def _feed_model(self, dataloader, train=False) -> dict:
        """ Feed the model with the data from the dataloader to train/validate/test the model

        Args:
            dataloader (): The DataLoader to feed the model with
            train (bool, optional): Whether the model is training. Defaults to False if validating or testing.

        Returns:
            dict: A dictionary containing the performance metrics for this epoch, including a list of batch metrics.
        """
        
        # Initialise metrics for whole epic (total)
        loss_crop_total = 0
        loss_state_total = 0
        loss_combined_total = 0
        correct_crop_total = 0
        correct_state_total = 0
        count_total = 0
        
        metrics = {
            'batches': [],
        }
        
        # Create the batch progress bar
        batch_progress = tqdm(enumerate(dataloader), desc="Batch", leave=False, total=len(dataloader))
        
        # Start feeding the model in batches
        for i, (images, crop_labels, state_labels) in batch_progress:
            
            # If training, set the optimiser to zero the gradients because we are about to calculate new gradients
            if train:
                self.optimiser.zero_grad()
                
            # Move data to GPU if available
            images = images.to(self.model.device)
            crop_labels = crop_labels.to(self.model.device)
            state_labels = state_labels.to(self.model.device)
            
            # Forward pass
            crop_predictions, state_predictions = self.model(images)
            
            # Calculate the loss
            loss_crop_batch = self.crop_criterion(crop_predictions, crop_labels)
            loss_state_batch = self.state_criterion(state_predictions, state_labels)
            loss_combined_batch = loss_crop_batch + loss_state_batch

            if train:
                # Backward pass
                loss_crop_batch.backward(retain_graph=True)
                loss_state_batch.backward()
                self.optimiser.step()
                
            # Calculate correct predictions
            _, predicted_crop = torch.max(crop_predictions, 1)
            _, predicted_state = torch.max(state_predictions, 1)
            correct_crop_batch = (predicted_crop == crop_labels).sum().item()
            correct_state_batch = (predicted_state == state_labels).sum().item()
            count_batch = crop_labels.size(0)
            
            # Capture batch metrics
            batch_metrics = {
                'loss': {
                    'crop': loss_crop_batch.item(),
                    'state': loss_state_batch.item(),
                    'combined': loss_combined_batch.item(),
                },
                'accuracy': {
                    'crop': correct_crop_batch / count_batch,
                    'state': correct_state_batch / count_batch,
                    'average': ((correct_crop_batch / count_batch) + (correct_state_batch / count_batch)) / 2
                },
                'correct': {
                    'crop': correct_crop_batch,
                    'state': correct_state_batch,
                },
                'count': count_batch
            }
            metrics['batches'].append([i, batch_metrics])
            
            # Update total epoch metrics
            loss_crop_total += loss_crop_batch.item()
            loss_state_total += loss_state_batch.item()
            loss_combined_total += loss_combined_batch.item()
            correct_crop_total += correct_crop_batch
            correct_state_total += correct_state_batch
            count_total += count_batch
            
            batch_progress.set_postfix(self._format_metrics(batch_metrics))
            
        metrics['loss'] = {
            'crop': loss_crop_total / len(dataloader),
            'state': loss_state_total / len(dataloader),
            'combined': loss_combined_total / len(dataloader)
        }
        
        metrics['accuracy'] = {
            'crop': correct_crop_total / count_total,
            'state': correct_state_total / count_total,
            'average': ((correct_crop_total / count_total) + (correct_state_total / count_total)) / 2
        }
        
        metrics['correct'] = {
            'crop': correct_crop_total,
            'state': correct_state_total
        }
        
        metrics['count'] = count_total
        
        return metrics
                
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
    
    def _calculate_change(self, new_value: float, old_value: float) -> float:
        """ Calculate the percentage change between two values

        Args:
            new_value (float): The value it is changing to
            old_value (float): The value it is changing from

        Returns:
            float: The percentage change between the two values
        """
        if old_value == 0:
            return 0
        return new_value / old_value - 1
    
    
    def _is_better_metric(self, metric) -> bool:
        """ Determine if the current epoch is better than the best epoch based on the average change percentage of the loss and accuracy metrics

        Args:
            metric (dict): The progression metrics for the current epoch

        Returns:
            bool: True if the current epoch is better than the best epoch, False otherwise
        """
        
        # Calculate the change percentage for each loss metric from the best metric to the currently observed metric set
        try:
            val_loss_crop_change = self._calculate_change(metric['loss']['crop'], self.performance_metrics['val']['loss']['crop'])
            val_loss_state_change = self._calculate_change(metric['loss']['state'], self.performance_metrics['val']['loss']['state'])
        except KeyError as e:
            logger.error(f"Metrics dictionary should have the keys 'loss' and 'accuracy' - {e}.")
            raise ValueError(f"Metrics dictionary should have the keys 'loss' and 'accuracy' - {e}")
        # Get the average loss change percentage, convert to absolute value for better comparison
        average_loss_change = abs((
            val_loss_crop_change +
            val_loss_state_change
        ) / 4)
        
        # Calculate the change percentage for each loss metric from the best metric to the currently observed metric set
        try:
            val_accuracy_crop_change = self._calculate_change(metric['accuracy']['crop'], self.performance_metrics['val']['accuracy']['crop'])
            val_accuracy_state_change = self._calculate_change(metric['accuracy']['state'], self.performance_metrics['val']['accuracy']['state'])
        except KeyError as e:
            logger.error(f"Metrics dictionary should have the keys 'loss' and 'accuracy' - {e}.")
            raise ValueError(f"Metrics dictionary should have the keys 'loss' and 'accuracy' - {e}")

        # Get the average accuracy change percentage, convert to absolute value for better comparison
        average_accuracy_change = abs((
            val_accuracy_crop_change +
            val_accuracy_state_change
        ) / 4)
        
        # Get the average of the average loss and accuracy change
        average = (average_loss_change + average_accuracy_change) / 2
    
        # If the average is greater than 0, then the current epoch is better than the best epoch
        if average > 0:
            return True
        
        else:
            return False
            
    def _format_metrics(self, metrics, dataset_names: list = None) -> dict:
        """ Format the metrics into a dictionary of formatted strings

        Args:
            metrics (list or dict): A list of metric dictionaries or a single metric dictionary
            dataset_names (list, optional): A list of dataset names to combine the metrics with. Defaults to None if a single metric dictionary is provided.

        Raises:
            ValueError: Raised when the metrics and dataset_names lists are not the same length
            ValueError: Raised when the metrics dictionary does not have the keys 'loss' and 'accuracy'

        Returns:
            dict: A dictionary of formatted metric strings
        """

        if isinstance(metrics, list) and isinstance(dataset_names, list) and len(metrics) == len(dataset_names):
            formatted_metrics = {}
            zipped_metrics = zip(metrics, dataset_names)
            for i, (metric, dataset_name) in enumerate(zipped_metrics):
                letter = dataset_name[i][0].upper()
                
                try:
                    formatted_metrics[f'{letter}LC'] = self._format_loss(metric['loss']['crop'])
                    formatted_metrics[f'{letter}LS'] = self._format_loss(metric['loss']['state'])
                    formatted_metrics[f'{letter}LA'] = self._format_accuracy(metric['accuracy']['crop'])
                    formatted_metrics[f'{letter}SA'] = self._format_accuracy(metric['accuracy']['state'])
                except KeyError as e:
                    logger.error(f"Metrics dictionary should have the keys 'loss' and 'accuracy' - {e}")
                    raise ValueError(f"Metrics dictionary should have the keys 'loss' and 'accuracy' - {e}")
            return formatted_metrics
        
        elif isinstance(metrics, dict):
            try:
                return {
                    'CL': self._format_loss(metrics['loss']['crop']),
                    'SL': self._format_loss(metrics['loss']['state']),
                    'CA': self._format_accuracy(metrics['accuracy']['crop']),
                    'SA': self._format_accuracy(metrics['accuracy']['state'])
                }
            except KeyError as e:
                logger.error(f"Metrics dictionary should have the keys 'loss' and 'accuracy' - {e}")
                raise ValueError(f"Metrics dictionary should have the keys 'loss' and 'accuracy' - {e}")
        else:
            logger.error("Metrics and combined lists should be the same length, or metrics should be a dictionary")
            raise ValueError("Metrics and combined lists should be the same length, or metrics should be a dictionary")
            
    def _format_accuracy(self, metric: float) -> str:
        """ Format the accuracy metric into a percentage string

        Args:
            metric (float): The accuracy metric to format

        Returns:
            str: The formatted accuracy metric as a percentage string
        """
        return f"{metric * 100:.1f}%"
    
    def _format_loss(self, metric: float) -> str:
        """ Format the loss metric to 3 decimal places

        Args:
            metric (float): The loss metric to format

        Returns:
            str: The formatted loss metric to 3 decimal places
        """
        return f"{metric:.3f}"
    
    def _get_dataloader(self, dataset, shuffle=True):
        """ Get a DataLoader for the dataset

        Args:
            dataset (torch.utils.data.Dataset): The dataset to create the DataLoader for
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.

        Returns:
            torch.utils.data.DataLoader: The DataLoader for the dataset 
        """
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.pipeline_config['training']['batch_size'],
            shuffle=shuffle,
            num_workers=self.pipeline_config['training']['num_workers'],
            pin_memory=True)
        return dataloader


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
    pipeline.save_pipeline()
    
    # Save Confusion matrix metrics
    pipeline.save_confusion_matrix()
    
    # Save the evaluation graphs
    pipeline.save_graphs()
    
    # Package the pipeline for API return
    pipeline_package = pipeline.package()
    
    logger.info("Training pipeline completed")
    
    return pipeline_package
