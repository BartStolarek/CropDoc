from CropDoc.app.service.data import ResNet50V2DatasetManager, ResNet50V2ConcatDatasetManager

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
import os

# Example usage:
config = {
    'model_name': 'resnet50',  # or 'xception', 'vgg16', 'yolov5'
    'num_classes': 10,
    'learning_rate': 0.001,
    'num_epochs': 10,
    'batch_size': 32,
    'output_dir': './output',
    # Add other configuration parameters as needed
}




class BasePipeline:
    def __init__(self, config,):
        self.config = config
        
        if 'model_name' not in self.config.keys():
            raise ValueError("Model name not specified in config")
        
        if 'learning_rate' not in self.config.keys():
            raise ValueError("Learning rate not specified in config")
        
        # Setup the device to use for training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialise the model
        self.model = self._create_model()
        
        # Move the model to the device
        self.model = self.model.to(self.device)
        
        # Initialise the loss function
        self.criterion = self._get_loss_function()
        
        # Initialise the optimiser
        self.optimizer = self._get_optimizer()
        

    def _create_model(self):
        model_name = self.config['model_name'].lower()

        if model_name == 'xception':
            pass
        elif model_name == 'resnet50':
            model = self._create_resnet50_model(number_of_crop_classes=None, number_of_state_classes=None)
        elif model_name == 'vgg16':
            pass
        elif model_name == 'yolov5':
            pass
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        return model.to(self.device)

    def _create_resnet50_model(self, number_of_crop_classes, number_of_state_classes):
        
        # Load the pre-trained ResNet50 model
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Modify the model to have the correct number of output classes
        number_of_features = model.fc.in_features
        model.fc = nn.Identity()  # Replace the fully connected layer with an identity layer
        model.fc_crop = nn.Linear(number_of_features, number_of_crop_classes)
        model.fc_state = nn.Linear(number_of_features, number_of_state_classes)
        
        return model
    
    def _get_loss_function(self):
        return nn.CrossEntropyLoss()  # Default loss, override if needed

    def _get_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])

    def _get_data_loaders(self):
        pass
    
    def load_datasets(self):
        

    def train_epoch(self):
        pass

    def validate(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def save_model(self):
        os.makedirs(self.config['output_dir'], exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.config['output_dir'], f"{self.config['model_name']}.pth"))
        print(f"Model saved to {self.config['output_dir']}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
        
    def evaluate_model(self):
        pass
    
    def plot_class_frequency_histogram(self):
        pass
        
    def plot_confusion_matrix(self):
        pass

pipeline = BasePipeline(config)
pipeline.train()
pipeline.test()
pipeline.save_model()


class ResNet50V2Pipeline():
    
    def __init__(self, crops: list, transformers: dict):
        """ Initialize the ResNet50V2 pipeline with the specified crops and transformers.
        
        Args:
            crops (list): A list of crops to load the dataset for.
            transformers (dict): A dictionary of torchvision.transformers.Compose with transformers within for the training and test sets.
        """
        self.crops = crops
        self.transformers = transformers
        self.dataset_manager = ResNet50V2DatasetManager(crops, transformers)
        self.concat_dataset_manager = ResNet50V2ConcatDatasetManager(crops, transformers)
        
    def load_datasets(self):
        """ Load the ResNet50V2 datasets for the specified crops and transformers. """
        self.train_datasets, self.test_datasets, self.all_classes = self.dataset_manager.load_datasets()
        
    def load_concat_datasets(self):
        """ Load the ResNet50V2 concatenated datasets for the specified crops and transformers. """
        self.train_dataset, self.test_dataset, self.all_classes = self.concat_dataset_manager.load_concat_datasets()
        
    def load_model(self):
        """ Load the ResNet50V2 model. """
        self.model = ResNet50V2()
        
    def train_model(self):
        """ Train the ResNet50V2 model. """
        self.model.train(self.train_datasets)
        
    def test_model(self):
        """ Test the ResNet50V2 model. """
        self.model.test(self.test_datasets)
        
    def save_model(self):
        """ Save the ResNet50V2 model. """
        self.model.save()
        
    def evaluate_model(self):
        """ Evaluate the ResNet50V2 model. """
        self.model.evaluate()
        
    def plot_class_frequency_histogram(self):
        """ Plot the class frequency histogram for the ResNet50V2 dataset. """
        self.model.plot_class_frequency_histogram()
        
    def plot_confusion_matrix(self):
        """ Plot the confusion matrix for the ResNet50V2 model. """
        self.model.plot_confusion_matrix()
        
    def run(self):
        """ Run the ResNet50V2 pipeline. """
        self.load_datasets()
        self.load_model()
        self.train_model()
        self.test_model()
        self.save_model()
        self.evaluate_model()
        self.plot_class_frequency_histogram()
        self.plot_confusion_matrix()
        
    def run_concat(self):
        """ Run the ResNet50V2 concatenated pipeline. """
        self
        
        
        