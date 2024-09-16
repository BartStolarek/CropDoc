import numpy as np
import os
import torch
from app.pipeline_helper.model import ResNet50
from app.pipeline_helper.datasetadapter import Structure
from app.pipeline_helper.numpyarraymanager import NumpyArrayManager
from app.pipeline_helper.model import ModelMeta
from app.pipeline_helper.metricstrackers import ProgressionMetrics, PerformanceMetrics
from loguru import logger
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

MODEL_CLASSES = {
    'ResNet50': ResNet50,
    # 'VGG16': VGG16,
    # 'Xception': Xception,
}


class ModelManager:
    
    def __init__(self, config, output_directory, data_structure: Structure, eval=False):
        self.config = config
        
        self.name = config['pipeline']['model']
        self.version = config['pipeline']['version']
        self.name_version = f"{self.name}-{str(self.version)}"
        self.output_directory = output_directory
        self.eval = eval
        self.data_structure = data_structure
        
        self.model_path = os.path.join(output_directory, 'model', f"{self.name_version}.pth")
        self.meta_path = os.path.join(output_directory, 'model', f"{self.name_version}.meta")
        self.checkpoint_directory = os.path.join(output_directory, 'checkpoints')
        
        # Load existing model
        self.model, self.model_meta = self.load_model()
        
        # If no model was found, create a new model
        if self.model is None or self.model_meta is None:
            self.model, self.model_meta = self.create_model()
        
        # If model's structure (classes) is not consistent with dataset structure (classes)
        if not self.consistent_classes_with_dataset(data_structure):
            logger.info("Model structure is not consistent with data structure for label's; crop and state")
            
            # Check if dataset structure is subset of model structure
            if self.is_subset(subset=data_structure, superset=self.model_meta):
                logger.info("Data structure is a subset of model structure, continuing without replacing head")
            else:
                logger.warning("Data structure is not a subset of model structure, meaning the data structure contains classes not present in the model, which will require replacing the models head")
                self.replace_head()

            
        self.save_model()
    
    
    
    def is_subset(self, subset: Structure, superset: ModelMeta) -> bool:
        numpy_array_manager = NumpyArrayManager()
        if not numpy_array_manager.check_arrays_are_subset(subset.crops, superset.crops):
            logger.info(f"Data structure crops ({len(subset.crops)}) are not a subset of model crops ({len(superset.crops)})")
            return False
        
        if not numpy_array_manager.check_arrays_are_subset(subset.states, superset.states):
            logger.info(f"Data structure states ({len(subset.states)}) are not a subset of model states ({len(superset.states)})")
            return False
        
        logger.info(f"Data structure is a subset of model structure")
        return True
    
    def load_model(self):
        if os.path.exists(self.meta_path) and os.path.exists(self.model_path):
            model_meta = ModelMeta(torch.load(self.meta_path, weights_only=False))
            model = MODEL_CLASSES[self.name](len(model_meta.crops), len(model_meta.states))
            
            if self.eval:
                model.load_state_dict(torch.load(self.model_path, map_location='cpu', weights_only=False))
                model.eval()
                logger.info('Model loaded in evaluation mode')
            else:
                model.load_state_dict(torch.load(self.model_path, weights_only=False))
                model.train()
                logger.info('Model loaded in training mode')
            logger.info(f"Loaded model {self.name_version} with data structure crops: {len(model_meta.crops)} and states: {len(model_meta.states)}")
            logger.info(f"Model meta loaded: {model_meta}")
            logger.debug(f"Model has performance metrics: {model_meta.performance_metrics}")
            return model, model_meta
        else:
            logger.info(f"Model not found at {self.model_path} or Model meta not found at {self.meta_path}")
            return None, None
            
    def create_model(self):
        model = MODEL_CLASSES[self.name](len(self.data_structure.crops), len(self.data_structure.states))
        model_meta = ModelMeta({
            'epochs': 0,
            'crops': self.data_structure.crops,
            'states': self.data_structure.states,
            'name': self.config['pipeline']['model'],
            'version': self.config['pipeline']['version'],
            'progression_metrics': ProgressionMetrics(),
            'performance_metrics': PerformanceMetrics()
        })
        logger.info(f"Created new model {self.name_version} with data structure crops: {len(self.data_structure.crops)} and states: {len(self.data_structure.states)}")
        if self.eval:
            logger.info("Model set to evaluation mode")
            model.eval()
        else:
            model.train()
            logger.info("Model set to training mode")
        return model, model_meta
    
    def consistent_classes_with_dataset(self, data_structure: Structure) -> bool:
        
        numpy_array_manager = NumpyArrayManager()
        
        if not numpy_array_manager.check_arrays_are_exact(self.model_meta.crops, data_structure.crops):
            logger.info(f"Model crops ({len(self.model_meta.crops)}) are not the same as data structure crops ({len(data_structure.crops)})")
            return False
        
        if not numpy_array_manager.check_arrays_are_exact(self.model_meta.states, data_structure.states):
            logger.info(f"Model states ({len(self.model_meta.states)}) are not the same as data structure states({len(data_structure.states)})")
            return False
        
        logger.info(f"Model classes are consistent with dataset classes")
        return True
    
    def replace_head(self):
        user_input = input("You are about to replace the head of the model, are you sure you want to continue? (Y/N): ")
        if user_input == 'N':
            raise Exception("User chose not to replace the head of the model. Exiting...")
        elif user_input == 'Y':
            self.model.create_new_head(len(self.data_structure.crops), len(self.data_structure.states))
            self.model_meta.crops = self.data_structure.crops
            self.model_meta.states = self.data_structure.states
            logger.info(f"Replaced head of model {self.name_version} with crops: {len(self.data_structure.crops)} and states: {len(self.data_structure.states)}")
            logger.warning("Model will require training, as the head has been replaced")
        else:
            raise Exception("Invalid input. Exiting...")
    
    def save_model(self):
        if not os.path.exists(os.path.join(self.output_directory, 'model')):
            os.makedirs(os.path.join(self.output_directory, 'model'))
        torch.save(self.model.state_dict(), self.model_path)
        torch.save(self.model_meta.to_dict(), self.meta_path)
        logger.info(f"Model saved at {self.model_path}")
        logger.info(f"Model meta saved at {self.meta_path}")
        
    def get_model(self):
        return self.model
    
    def get_number_of_trained_epochs(self):
        return self.model_meta.epochs

    def get_progress_metrics_tracker(self):
        return self.model_meta.progression_metrics
    
    def get_performance_metrics_tracker(self):
        return self.model_meta.performance_metrics