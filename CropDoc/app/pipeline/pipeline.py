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
import shutil
import json
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from app.pipeline_helper.transformer import TransformerManager
from app.pipeline_helper.dataset import CropCCMTDataset, PlantVillageDataset, BaseDataset
from app.pipeline_helper.dataloader import TransformDataLoader
from app.pipeline_helper.model import MultiHeadResNetModel

from pprint import pprint

warnings.filterwarnings("ignore",
                        category=FutureWarning,
                        module="torch.nn.parallel.parallel_apply")


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
    
    def test_model(self):
        
        # Load the saved trained pipeline
        pipeline_path = os.path.join(self.pipeline_output_dir, 'final', f'{self.pipeline_name}-{self.pipeline_version}.pth') # Get the path of the final pipeline
        pipeline = torch.load(pipeline_path, weights_only=False) # Load the pipeline
        
        # Load the metrics
        self.progression_metrics = pipeline['progression_metrics']
        self.performance_metrics = pipeline['performance_metrics']
        
        # Load the transformer manager to handle loading transformers from config
        self.transformer_manager = TransformerManager(
            self.pipeline_config['data']['transformers']  # TODO: Add to report that we are using transformers to augment the training data
        )
        
        # Load the test dataset
        dataset_kwargs = {
            'dataset_path': self.dataset_root,
            'split': 'test',
            'crop_index_map': pipeline['crop_index_map'],
            'state_index_map': pipeline['state_index_map']
        }
        
        self.test_data = self._get_dataset_object(**dataset_kwargs)
        
        # If in development dataset needs to be reduced for faster training
        development_reduction = self.pipeline_config['data']['reduce']
        if development_reduction < 1:
            self.test_data.equal_reduce(development_reduction)
            
        self.crop_index_map = self.test_data.crop_index_map
        self.state_index_map = self.test_data.state_index_map
    
        # Initialise a model
        self.model = MultiHeadResNetModel( # Create a new model
            num_classes_crop=self.test_data.get_unique_crop_count(),
            num_classes_state=self.test_data.get_unique_state_count()
        )
        
        # Load the trained model weights from the pipeline into the new model.
        self.model.load_state_dict(torch.load(pipeline_path, weights_only=False)['model_state_dict'])
        
        # Set the model to eval mode
        self.model.eval()
        
        # Define the test dataloader
        test_dataloader = self._get_dataloader(self.test_data, split='test', shuffle=False)
        
        # Set the criterion for the loss function
        self.crop_criterion = torch.nn.CrossEntropyLoss()
        self.state_criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            
            # Feed the model the test data
            test_metrics = self._feed_model(test_dataloader, capture_batches=False)
            test_name = 'test' + '_' + self.pipeline_config['dataset_class']
            self.performance_metrics[test_name] = test_metrics
            
        logger.info(f'Test Results:\n {test_metrics}')
        
        return test_metrics
        
    def predict_model(self, image_path: str) -> dict:
        """ Predict the crop and state of the image at the provided path

        Args:
            image_path (str): The path to the image file, suggest using data/tmp folder

        Returns:
            dict: A dictionary containing the prediction results for the crop and state
        """
        
        
        # Load the saved trained pipeline
        pipeline_path = os.path.join(self.pipeline_output_dir, 'final', f'{self.pipeline_name}-{self.pipeline_version}.pth') # Get the path of the final pipeline
        pipeline = torch.load(pipeline_path, weights_only=False) # Load the pipeline
        
        # Load the transformer manager to handle loading transformers from config
        self.transformer_manager = TransformerManager(
            self.pipeline_config['data']['transformers']  # TODO: Add to report that we are using transformers to augment the training data
        )
        
        self.crop_index_map = pipeline['crop_index_map']
        self.state_index_map = pipeline['state_index_map']
        
        # Initialise a model
        self.model = MultiHeadResNetModel( # Create a new model
            num_classes_crop=len(pipeline['crop_index_map']),
            num_classes_state=len(pipeline['state_index_map'])
        )
        
        try:
            # Load the trained model weights from the pipeline into the new model. 
            self.model.load_state_dict(torch.load(pipeline_path, weights_only=False)['model_state_dict'])
        except RuntimeError as e:
            logger.error(f"Error loading model, shape is wrong. Potential issues is that the config file's data reduce value is different from what was trained on. {e}")
            raise RuntimeError(f"Error loading model, shape is wrong. Potential issues is that the config file's data reduce value is different from what was trained on. {e}")
        
        
        # Set the model to eval mode
        self.model.eval()
        
        # Load transformers
        self.transformer_manager = TransformerManager(
            self.pipeline_config['data']['transformers']
        )
        
        # Load the image and apply transformers
        image = self._load_image(image_path, split='test')
        
        # Convert PIL Image to PyTorch tensor
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
        ])
        image = transform(image)
        
        # Unsqueeze the image to add a batch dimension
        image = image.unsqueeze(0).to(self.model.device)
        
        # Feed the image to the model and make a prediction
        with torch.no_grad():
            crop_predictions, state_predictions = self.model(image) # Get the pytorch tensors (multi dimensional arrays) with raw scores (not probabilities), shape: (1, num_classes)
            
        # Get the predicted classes
        crop_confidence, predicted_crop = torch.max(crop_predictions, 1) # Tensor with highest probability class, shape: (1), Tensor with index of highest probability class, shape: (1)
        state_confidence, predicted_state = torch.max(state_predictions, 1)
        
        # Apply softmax to get probabilities
        crop_probabilities = torch.nn.functional.softmax(crop_predictions, dim=1) # Tensor with probabilities for each class summing to 1, shape: (1, num_classes)
        state_probabilities = torch.nn.functional.softmax(state_predictions, dim=1) # Tensor with probabilities for each class summing to 1, shape: (1, num_classes)
        
        # Convert to numpy for easier handling
        crop_confidence = crop_confidence.cpu().numpy() 
        state_confidence = state_confidence.cpu().numpy()
        predicted_crop = predicted_crop.cpu().numpy()
        predicted_state = predicted_state.cpu().numpy()
        
        # Get the probability of the predicted class
        crop_probability = torch.max(crop_probabilities).item()
        state_probability = torch.max(state_probabilities).item()
        
        # Apply confidence threshold
        crop_mask = crop_probability > self.pipeline_config['prediction']['crop']['confidence_threshold']
        state_mask = state_probability > self.pipeline_config['prediction']['state']['confidence_threshold']

        # Get the crop class prediction and confidence
        crop_prediction = self.crop_index_map[predicted_crop[0]] if crop_mask else 'Unknown'
        state_prediction = self.state_index_map[predicted_state[0]] if state_mask else 'Unknown'
        
        # Get class names for all classes
        crop_class_names = [self.crop_index_map[i] for i in range(len(self.crop_index_map))]
        state_class_names = [self.state_index_map[i] for i in range(len(self.state_index_map))]

        # Map probabilities to class names
        crop_probabilities_dict = dict(zip(crop_class_names, crop_probabilities.squeeze().cpu().numpy()))
        state_probabilities_dict = dict(zip(state_class_names, state_probabilities.squeeze().cpu().numpy()))

        result = {
            'crop': {
                'prediction': crop_prediction,
                'confidence': crop_confidence[0],
                'probability': crop_probability,
                'confidence_threshold': self.pipeline_config['prediction']['crop']['confidence_threshold'],
                'class_probabilities': crop_probabilities_dict
            },
            'state': {
                'prediction': state_prediction,
                'confidence': state_confidence[0],
                'probability': state_probability,
                'confidence_threshold': self.pipeline_config['prediction']['state']['confidence_threshold'],
                'class_probabilities': state_probabilities_dict
            }
        }
        
        logger.info(f"Prediction for image {image_path}:\n{result}")
        
        return result
    
    def save_prediction(self, prediction: dict):
        """ Save the prediction to a file in the predictions folder """
        
        logger.debug("Saving prediction")
        
        # Get the prediction folder within the pipeline folder
        prediction_folder = os.path.join(self.pipeline_output_dir, 'predictions')
        
        # Ensure the directory exists
        os.makedirs(prediction_folder, exist_ok=True)
        
        # Get the current time to use as a unique identifier
        current_time = time.strftime("%Y%m%d-%H%M%S")
        
        # Save the prediction to a file
        prediction_file = os.path.join(prediction_folder, f'{self.pipeline_name}-{self.pipeline_version}-{current_time}.npy')
        
        # Save the prediction using NumPy
        np.save(prediction_file, prediction)
        
        logger.info(f"Saved prediction to {prediction_file}")
        
        # Optionally, save a human-readable text version
        text_file = os.path.join(prediction_folder, f'{self.pipeline_name}-{self.pipeline_version}-{current_time}.txt')
        with open(text_file, 'w') as f:
            f.write(str(prediction))
        
        logger.info(f"Saved human-readable prediction to {text_file}")
        
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
        
        # Load the train dataset
        dataset_kwargs = {
            'dataset_path': self.dataset_root,
            'split': 'train'
        }
       
        
        # Initialise the training dataset     
        self.train_data = self._get_dataset_object(**dataset_kwargs)
        
        if pipeline_exists:
            # Load the crop and state index map from the saved file
            crop_index_map = torch.load(self.pipeline_path, weights_only=False)['crop_index_map']
            state_index_map = torch.load(self.pipeline_path, weights_only=False)['state_index_map']

            # Give it to the dataset class to keep mapping the same
            dataset_kwargs['crop_index_map'] = crop_index_map
            dataset_kwargs['state_index_map'] = state_index_map
        
        # Resave the crop and state index maps
        self.crop_index_map = self.train_data.crop_index_map
        self.state_index_map = self.train_data.state_index_map
        
        logger.debug(f"Pipeline\nCrop index map: {self.crop_index_map}")
        logger.debug(f"Pipeline\nState index map: {self.state_index_map}")
        
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
            self.model.load_state_dict(torch.load(self.pipeline_path, weights_only=False)['model_state_dict'])
        
        # Define the loss functions for both heads
        self.crop_criterion = torch.nn.CrossEntropyLoss()  # TODO: Add to report that we are using CrossEntropyLoss for the loss functions for both heads
        self.state_criterion = torch.nn.CrossEntropyLoss()
        
        # Define the optimiser
        self.optimiser = torch.optim.Adam(  # TODO: Add to the report that we are using Adam optimiser, and the learning rate it starts at
            self.model.parameters(),
            lr=self.pipeline_config['training']['learning_rate']
        )
        if pipeline_exists:
            self.optimiser.load_state_dict(torch.load(self.pipeline_path, weights_only=False)['optimiser_state_dict'])
        
        # Define the learning rate scheduler, which will reduce the learning rate when the model stops improving so that it can find a better minima
        active_scheduler = self.pipeline_config['training']['lr_scheduler']['active']  # Obtain the active scheduler set in the config
        scheduler_config = self.pipeline_config['training']['lr_scheduler'][active_scheduler]  # Obtain the configuration for the active scheduler
        scheduler_object = getattr(torch.optim.lr_scheduler, active_scheduler)  # TODO: Add to the report the scheduler we are using (from config file)
        self.scheduler = scheduler_object(
            self.optimiser,
            **scheduler_config
        )
        if pipeline_exists:
            self.scheduler.load_state_dict(torch.load(self.pipeline_path, weights_only=False)['scheduler_state_dict'])
        
        logger.info('Pipeline Initialisation Complete')
        
        logger.info('\nTraining loop index:\n' +
                    "T: Train, V: Validation\n" +
                    "C: Crop, S: State\n" +
                    "L: Loss, A: Accuracy"
                    )
        
        # Define the best validation loss
        checkpoint_interval = self.pipeline_config['training']['checkpoint_interval']  # The minimum epochs to go buy before checking to save another checkpoint

        # Define the number of epochs to train for and set the epoch range
        self.epochs = self.pipeline_config['training']['epochs']
        if pipeline_exists:
            checkpoint = torch.load(self.pipeline_path, weights_only=False)
            start = checkpoint['epochs'] + 1
            end = start + self.epochs
            self.total_pretrained_epochs = checkpoint['epochs']
            logger.info(f'Starting training loop for epochs {start} to {end - 1} ({end - start}). Total pre-trained epochs: {self.total_pretrained_epochs}')
        else:
            start = 1
            end = self.epochs + 1
            logger.info(f'Starting training loop for epochs {start} to {end - 1}')
            
        # Initialise epoch progress bar and training metrics list
        self.terminal_width = shutil.get_terminal_size().columns
        epochs_progress = tqdm(range(start, end), desc="Epoch", leave=True, ncols=int(self.terminal_width * 0.99))
        
        if pipeline_exists:
            self.progression_metrics = torch.load(self.pipeline_path, weights_only=False)['progression_metrics']
            self.performance_metrics = torch.load(self.pipeline_path, weights_only=False)['performance_metrics']
            
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
            
        if pipeline_exists:
            self.total_pretrained_epochs += self.epochs
        else:
            self.total_pretrained_epochs = self.epochs
            
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
        if epoch:
            directory = os.path.join(self.pipeline_output_dir, 'checkpoints')
            file_name = f'{self.pipeline_name}-{self.pipeline_version}-epoch-{epoch}.pth'
            save_dict['epochs'] = epoch
        else:
            directory = os.path.join(self.pipeline_output_dir, 'final')
            file_name = f'{self.pipeline_name}-{self.pipeline_version}.pth'
            save_dict['epochs'] = self.total_pretrained_epochs
            
            
        
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
        train_dataloader = self._get_dataloader(dataset=train_dataset, shuffle=True, split='train')
        val_dataloader = self._get_dataloader(dataset=val_dataset, shuffle=False, split='val')
        
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
    
    def _get_dataset_object(self, **kwargs) -> BaseDataset:
        """ Get the dataset object based on the dataset class specified in the config file

        Raises:
            ValueError: If the dataset class is invalid

        Returns:
            BaseDataset: The dataset object
        """
        match self.pipeline_config['dataset_class']:
            case 'CropCCMTDataset':
                return CropCCMTDataset(**kwargs)
            case 'PlantVillageDataset':
                return PlantVillageDataset(**kwargs)
            case _:
                logger.error(f"Invalid dataset class: {self.pipeline_config['dataset_class']}, please update the config file")
                raise ValueError(f"Invalid dataset class: {self.pipeline_config['dataset_class']}, please update the config file")
      
    def _feed_model(self, dataloader, train=False, capture_batches=True) -> dict:
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
        
        all_crop_predictions = []
        all_crop_labels = []
        all_state_predictions = []
        all_state_labels = []
        
        metrics = {
            'batches': [],
        }
        
        # Create the batch progress bar
        self.terminal_width = shutil.get_terminal_size().columns
        batch_progress = tqdm(enumerate(dataloader), desc="Batch", leave=False, total=len(dataloader), ncols=int(self.terminal_width * 0.99))
        
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
            
            # Store predictions and labels for later confusion matrix
            all_crop_predictions.extend(predicted_crop.cpu().numpy())
            all_crop_labels.extend(crop_labels.cpu().numpy())
            all_state_predictions.extend(predicted_state.cpu().numpy())
            all_state_labels.extend(state_labels.cpu().numpy())
            
            # Capture batch metrics
            if capture_batches:
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
            
            if capture_batches:
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
        
        # Calculate confusion matrices
        crop_confusion_matrix = confusion_matrix(all_crop_labels, all_crop_predictions)
        state_confusion_matrix = confusion_matrix(all_state_labels, all_state_predictions)
        
        # Calculate precision, recall, f1 score
        crop_precision, crop_recall, crop_f1, crop_support = precision_recall_fscore_support(all_crop_labels, all_crop_predictions, average='weighted', zero_division=0)
        state_precision, state_recall, state_f1, state_support = precision_recall_fscore_support(all_state_labels, all_state_predictions, average='weighted', zero_division=0)
        
        # Add these metrics to your existing metrics dictionary
        metrics['confusion_matrix'] = {
            'crop': crop_confusion_matrix.tolist(),
            'state': state_confusion_matrix.tolist()
        }
        metrics['precision'] = {
            'crop': crop_precision,
            'state': state_precision
        }
        metrics['recall'] = {
            'crop': crop_recall,
            'state': state_recall
        }
        metrics['f1_score'] = {
            'crop': crop_f1,
            'state': state_f1
        }
        metrics['support'] = {
            'crop': crop_support,
            'state': state_support
        }
        
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
            for i in range(len(metrics)):
                letter = dataset_names[i][0].upper()
                
                try:
                    formatted_metrics[f'{letter}CL'] = self._format_loss(metrics[i]['loss']['crop'])
                    formatted_metrics[f'{letter}SL'] = self._format_loss(metrics[i]['loss']['state'])
                    formatted_metrics[f'{letter}CA'] = self._format_accuracy(metrics[i]['accuracy']['crop'])
                    formatted_metrics[f'{letter}SA'] = self._format_accuracy(metrics[i]['accuracy']['state'])
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
    
    def _get_dataloader(self, dataset, split, shuffle=True):
        """ Get a DataLoader for the dataset

        Args:
            dataset (torch.utils.data.Dataset): The dataset to create the DataLoader for
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.

        Returns:
            torch.utils.data.DataLoader: The DataLoader for the dataset 
        """
        
        dataloader = TransformDataLoader(
            dataset,
            batch_size=self.pipeline_config['training']['batch_size'],
            shuffle=shuffle,
            num_workers=self.pipeline_config['training']['num_workers'],
            pin_memory=True,
            transform=self.transformer_manager.transformers[split]
        )
        return dataloader

    def _load_image(self, image_path: str, split: str) -> Image:
        """ Load the image from the provided path and apply the transformers for the specified split

        Args:
            image_path (str): The path to the image file
            split (str): The dataset split that the image belongs to (train, val, test)

        Raises:
            ValueError: If the split is not one of 'train', 'val', or 'test'
            FileNotFoundError: If the image file is not found at the provided path

        Returns:
            Image: The loaded image with the transformers applied
        """
        # Check right split was provided
        if split not in ['train', 'val', 'test']:
            logger.error(f"Split must be one of 'train', 'val', or 'test', not {split}")
            raise ValueError(f"Split must be one of 'train', 'val', or 'test', not {split}")
        
        # Load the image
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError as e:
            logger.error(f"Image file not found at {image_path} - {e}")
            raise FileNotFoundError(f"Image file not found at {image_path} - {e}")
        
        # Apply the transformers
        image = self.transformer_manager.transformers[split](image)
        
        return image


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


def predict(config, image_path: str) -> dict:
    """ A method/function to run the prediction pipeline

    Args:
        config (dict): Config file that has been loaded as a dict
        image_path (str): A string pointing to the image in data/tmp

    Returns:
        dict: A dictionary containing the prediction results for the crop and state
    """
    
    logger.debug("Running the prediction pipeline")
     
    # Initialise the Pipeline
    pipeline = Pipeline(config)
    
    # Predict the image
    prediction = pipeline.predict_model(image_path)
    
    # Save prediction to a file
    pipeline.save_prediction(prediction)
    
    logger.info("Prediction pipeline completed")
    
    return prediction


def test(config):
    
    logger.debug("Running the test pipeline")
    
    # Initialise the Pipeline
    pipeline = Pipeline(config)
    
    # Test the model
    pipeline.test_model()
    
    # Package the pipeline for API return
    pipeline_package = pipeline.package()
    
    logger.info("Test pipeline completed")
    
    return pipeline_package
