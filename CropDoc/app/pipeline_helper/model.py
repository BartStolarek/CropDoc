import torch
import torchvision
import os
import numpy as np
from app.pipeline_helper.metricstrackers import PerformanceMetrics, ProgressionMetrics
from loguru import logger
from torch import nn

class ModelMeta:
    def __init__(self, meta_dict: dict):
        self.epochs = meta_dict['epochs']
        self.crops = np.array(meta_dict['crops'])
        self.states = np.array(meta_dict['states'])
        self.name = meta_dict['name']
        self.version = meta_dict['version']
        if isinstance(meta_dict['performance_metrics'], dict):
            self.performance_metrics = PerformanceMetrics(performance_dict=meta_dict['performance_metrics'])
        else:
            self.performance_metrics = meta_dict['performance_metrics']
        if isinstance(meta_dict['progression_metrics'], list):
            self.progression_metrics = ProgressionMetrics(performance_dict=meta_dict['progression_metrics'])
        else:
            self.progression_metrics = meta_dict['progression_metrics']
            
        logger.info(f"Initialised ModelMeta with epochs: {self.epochs}, crops: {len(self.crops)}, states: {len(self.states)}, name: {self.name}, version: {self.version}")
   
    def to_dict(self):
        if self.epochs is not None and not isinstance(self.epochs, int):
            raise ValueError(f"Epochs must be an integer or None, not {type(self.epochs)}")
        
        if self.performance_metrics is not None and not isinstance(self.performance_metrics, PerformanceMetrics):
            raise ValueError(f"Performance metrics must be a PerformanceMetrics object or None, not {type(self.performance_metrics)}")
        
        if self.progression_metrics is not None and not isinstance(self.progression_metrics, ProgressionMetrics):
            raise ValueError(f"Progression metrics must be a ProgressionMetrics object or None, not {type(self.progression_metrics)}")
        
        return {
            'epochs': self.epochs,
            'crops': self.crops.tolist(),  # Convert NumPy array to list
            'states': self.states.tolist(),  # Convert NumPy array to list
            'name': self.name,
            'version': self.version,
            'performance_metrics': self.performance_metrics.to_dict(),
            'progression_metrics': self.progression_metrics.to_dict()
        }
    
    def __str__(self):
        return f"{self.name} ({self.version}) trained for {self.epochs} epochs, with crops: {len(self.crops)} and states: {len(self.states)}"


class ResNet50(nn.Module):
    """A multi-head ResNet model for the CropCCMT dataset

    Args:
        torch (nn.Module): The PyTorch module
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
        super(ResNet50, self).__init__()
        
        logger.info(f'Initialising ResNet50 model with {num_classes_crop} crop classes and {num_classes_state} state classes')
        
        self.create_new_head(num_classes_crop, num_classes_state)

        # Wrap only the resnet part in DataParallel
        self.resnet = nn.DataParallel(self.resnet)

    def create_new_head(self, num_classes_crop, num_classes_state):
        """ Create new heads for the crop and state heads

        Returns:
            nn.Module: The model with new heads
        """
        
        # Check if GPU is available
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Load the ResNet50 model with pre-trained weights
        self.resnet = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.DEFAULT
        )  # TODO: Add to report why we used ResNet50 default weights and benefits
        
        num_ftres = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        self.crop_fc = nn.Linear(num_ftres, num_classes_crop)
        self.state_fc = nn.Linear(num_ftres, num_classes_state)
        
        self.resnet = self.resnet.to(self.device)
        self.crop_fc = self.crop_fc.to(self.device)
        self.state_fc = self.state_fc.to(self.device)
    
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

    def save_checkpoint(self, epoch, optimiser, scheduler, model_meta, filename, checkpoint_directory):
        """ Save the model checkpoint

        Args:
            epoch (int): The current epoch
            optimizer (torch.optim): The optimizer
            loss (float): The loss
            filename (str): The filename
            checkpoint_directory (str): The directory to save the checkpoint
        """
        directory_path = os.path.join(checkpoint_directory, f'epoch_{epoch}')
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)

        # Save the model checkpoint
        torch.save(self.state_dict(), os.path.join(directory_path, f'{filename}.pth'))
        
        # Save the optimiser checkpoint
        torch.save(optimiser.state_dict(), os.path.join(directory_path, 'optimiser.pth'))
        
        # Save the scheduler checkpoint
        torch.save(scheduler.state_dict(), os.path.join(directory_path, 'scheduler.pth'))
        
        # Save the model meta data
        torch.save(model_meta.to_dict(), os.path.join(directory_path, f'{filename}.meta'))
        
class VGG16(nn.Module):
    """A multi-head VGG16 model for the CropCCMT dataset"""

    def __init__(self, num_classes_crop, num_classes_state):
        """
        Initialize a multi-head VGG16 model with:
        - A VGG16 backbone with pre-trained weights
        - A crop head
        - A state head
        
        Also move the model to the GPU if available

        Args:
            num_classes_crop (int): The number of unique classes for the crop head
            num_classes_state (int): The number of unique classes for the state head
        """
        super(VGG16, self).__init__()
        
        logger.info(f'Initializing VGG16 model with {num_classes_crop} crop classes and {num_classes_state} state classes')
        
        self.create_new_head(num_classes_crop, num_classes_state)

        # Wrap only the vgg part in DataParallel
        self.vgg = nn.DataParallel(self.vgg)

    def create_new_head(self, num_classes_crop, num_classes_state):
        """Create new heads for the crop and state heads"""
        
        # Check if GPU is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.classifier_num = (num_classes_crop, num_classes_state)
        self.vgg = nn.Sequential(
            self.convLayer(3, 64),
            self.convLayer(64, 64),
            nn.MaxPool2d((2, 2), (2, 2)),
            self.convLayer(64, 128),
            self.convLayer(128, 128),
            nn.MaxPool2d((2, 2), (2, 2)),
            self.convLayer(128, 256),
            self.convLayer(256, 256),
            self.convLayer(256, 256),
            nn.MaxPool2d((2, 2), (2, 2)),
            self.convLayer(256, 512),
            self.convLayer(512, 512),
            self.convLayer(512, 512),
            nn.MaxPool2d((2, 2), (2, 2)),
            self.convLayer(512, 512),
            self.convLayer(512, 512),
            self.convLayer(512, 512),
            nn.MaxPool2d((2, 2), (2, 2))
        )
        
        self.avgPool = nn.AdaptiveAvgPool2d((7, 7))
        dropout = 0.5
        self.classifier1 = self.classifier(num_classes_crop, dropout)
        self.classifier2 = self.classifier(num_classes_state, dropout)
        
    
    def forward(self, x):
        """
        Forward pass through the model, and return the tensors for the crop and state heads as a tuple

        Args:
            x (torch.Tensor): The input tensor, where x.shape is torch.Size(<batch_size>, <num_channels>, <height>, <width>)

        Returns:
            tuple: A tuple containing the crop and state tensors
        """
        x = x.to(self.device)  # Move input to GPU if available
        
        # Forward pass through the VGG backbone
        x = self.vgg(x)
        x = self.avgPool(x)
        x = torch.flatten(x, 1)
        if isinstance(self.classifier_num, tuple):
            class_1 = self.classifier1(x)
            class_2 = self.classifier2(x)
            return class_1, class_2
        else:
            x = self.classifier1(x)
            return x

    def save_checkpoint(self, epoch, optimiser, scheduler, model_meta, filename, checkpoint_directory):
        """
        Save the model checkpoint

        Args:
            epoch (int): The current epoch
            optimizer (torch.optim): The optimizer
            scheduler: The learning rate scheduler
            model_meta: The model metadata
            filename (str): The filename
            checkpoint_directory (str): The directory to save the checkpoint
        """
        import os

        directory_path = os.path.join(checkpoint_directory, f'epoch_{epoch}')
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)

        # Save the model checkpoint
        torch.save(self.state_dict(), os.path.join(directory_path, f'{filename}.pth'))
        
        # Save the optimizer checkpoint
        torch.save(optimiser.state_dict(), os.path.join(directory_path, 'optimizer.pth'))
        
        # Save the scheduler checkpoint
        torch.save(scheduler.state_dict(), os.path.join(directory_path, 'scheduler.pth'))
        
        # Save the model meta data
        torch.save(model_meta.to_dict(), os.path.join(directory_path, f'{filename}.meta'))    

    def convLayer(self, layer_in: int, layer_out: int) -> nn.Sequential:
        return nn.Sequential(nn.Conv2d(layer_in, layer_out, 3, 1, padding="same"), nn.BatchNorm2d(layer_out), nn.ReLU())
    
    def classifier(self, num_classes: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(self.linLayer(7 * 7 * 512, 4096, dropout), self.linLayer(4096, 1024, dropout), nn.Linear(1024, num_classes))

    def linLayer(self, layer_in: int, layer_out: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(nn.Linear(layer_in, layer_out), nn.ReLU(), nn.Dropout(dropout))