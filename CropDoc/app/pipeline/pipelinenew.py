from app.pipeline_helper.datasetadapter import CropCCMTDataset, PlantVillageDataset
from app.pipeline_helper.datasetmanager import DatasetManager
from app.pipeline_helper.modelmanager import ModelManager
from app.pipeline_helper.transformermanager import TransformerManager
from app.pipeline_helper.optimisermanager import OptimiserManager
from app.pipeline_helper.schedulermanager import SchedulerManager
from app.pipeline_helper.trainingmanager import TrainingManager
import os
import torch

class Pipeline:
    def __init__(self, config):
        self.config = config
        self.model = config['pipeline']['model']
        self.version = config['pipeline']['version']
        self.output_name = f"{self.model}-{str(self.version)}"
        self.output_directory = os.path.join('data/output', f"{self.output_name}")
        
    def train(self):
        
        # Datasets
        dataset_manager = DatasetManager(config=self.config, output_directory=self.output_directory)
        train_dataset = dataset_manager.get_train_dataset()
        
        # Transformers
        transformer_manager = TransformerManager()
        train_transformers = transformer_manager.get_train_transformers()
        val_transformers = transformer_manager.get_val_transformers()
        
        # Models
        model_manager = ModelManager(
            config=self.config,
            output_directory=self.output_directory,
            data_structure=dataset_manager.structure,
            eval=False
        )

        self.model = model_manager.get_model()
        
        # Loss Functions
        crop_criterion = torch.nn.CrossEntropyLoss()
        state_criterion = torch.nn.CrossEntropyLoss()
        
        # Optimisers
        optimiser = OptimiserManager(model=self.model, config=self.config, output_directory=self.output_directory).get_optimiser()
        
        # Learning Rate Scheduler
        scheduler = SchedulerManager(optimiser, config=self.config, output_directory=self.output_directory).get_scheduler()
        
        training_manager = TrainingManager(config=self.config, output_directory=self.output_directory, model=self.model, optimiser=optimiser, scheduler=scheduler, crop_criterion=crop_criterion, state_criterion=state_criterion, train_dataset=train_dataset, train_transformers=train_transformers, val_transformers=val_transformers)
        
    def test(self):
        pass

    def predict(self):
        pass