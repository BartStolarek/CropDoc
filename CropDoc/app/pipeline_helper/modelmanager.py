import os
import torch
from app.pipeline_helper.model import ResNet50
from loguru import logger

MODEL_CLASSES = {
    'ResNet50': ResNet50,
    # 'VGG16': VGG16,
    # 'Xception': Xception,
}

class ModelManager:
    def __init__(self, config, output_directory, num_classes_crop, num_classes_state, eval=False):
        self.config = config
        self.output_directory = output_directory
        self.model_name = config['pipeline']['model']
        self.version = config['pipeline']['version']
        self.model_final_path = os.path.join(output_directory, 'final', f"{self.model_name}-{str(self.version)}.pth")
        self.checkpoint_directory_path = os.path.join(output_directory, 'checkpoints')
        self.eval = eval
        
        self.num_classes_crop = num_classes_crop
        self.num_classes_state = num_classes_state
        
        self.model = self._load_or_create_model()
        if self.eval:
            self.model.eval()
        else:
            self.model.train()
            
        
        
    def _load_or_create_model(self):
        if os.path.exists(self.model_final_path):
            model = self._load_model()
        else:
            model = self._create_model()
            self.training_history = {}
        
        return model
    
    def _load_model(self):
        model = MODEL_CLASSES[self.model_name](self.num_classes_crop, self.num_classes_state)
        if self.eval:
            model.load_state_dict(torch.load(self.model_final_path, map_location='cpu'))
        else:
            model.load_state_dict(torch.load(self.model_final_path))
        logger.info(f"Model loaded from {self.model_final_path}")
        return model
    
    def _create_model(self):
        logger.info(f"Creating new {self.model_name} model")
        return MODEL_CLASSES[self.model_name](self.num_classes_crop, self.num_classes_state)
    
    def create_new_head(self, model):
        logger.info("Creating new heads for the model")
        model.create_new_head(self.num_classes_crop, self.num_classes_state)
    
    def save_model(self, is_best=False):
        if is_best:
            torch.save(self.model.state_dict(), self.model_final_path)
            torch.save()
            
        else:
            checkpoint_path = os.path.join(self.checkpoint_directory_path, f"{self.model_name}-{str(self.version)}_checkpoint.pth")
            torch.save(self.model.state_dict(), checkpoint_path)
    
    def get_model(self):
        return self.model