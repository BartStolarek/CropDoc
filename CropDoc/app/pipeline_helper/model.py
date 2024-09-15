import torch
import torchvision
import os
import numpy as np
from app.pipeline_helper.metricstrackers import PerformanceMetrics, ProgressionMetrics
from loguru import logger

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
         
    def to_dict(self):
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




class ResNet50(torch.nn.Module):
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
        super(ResNet50, self).__init__()
        logger.info(f'Initialising ResNet50 model with {num_classes_crop} crop classes and {num_classes_state} state classes')
        # Check if GPU is available
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Load the ResNet50 model with pre-trained weights
        self.resnet = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.DEFAULT
        )  # TODO: Add to report why we used ResNet50 default weights and benefits

        self.create_new_head(num_classes_crop, num_classes_state)

        # Wrap only the resnet part in DataParallel
        self.resnet = torch.nn.DataParallel(self.resnet)

    def create_new_head(self, num_classes_crop, num_classes_state):
        """ Create new heads for the crop and state heads

        Returns:
            torch.nn.Module: The model with new heads
        """
        num_ftres = self.resnet.fc.in_features
        
        self.crop_fc = torch.nn.Linear(num_ftres, num_classes_crop)
        self.state_fc = torch.nn.Linear(num_ftres, num_classes_state)
        
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
        print(f"Input shape: {x.shape}")
        
        # Forward pass through the ResNet backbone
        x = self.resnet(x)
        print(f"After ResNet shape: {x.shape}")

        # Forward pass through the crop and state heads
        crop_out = self.crop_fc(x)
        state_out = self.state_fc(x)

        print(f"Crop output shape: {crop_out.shape}")
        print(f"State output shape: {state_out.shape}")
        
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
        
        