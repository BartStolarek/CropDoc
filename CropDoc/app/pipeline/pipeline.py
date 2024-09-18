from app.pipeline_helper.datasetadapter import CropCCMTDataset, PlantVillageDataset
from app.pipeline_helper.datasetmanager import DatasetManager
from app.pipeline_helper.modelmanager import ModelManager
from app.pipeline_helper.transformermanager import TransformerManager
from app.pipeline_helper.optimisermanager import OptimiserManager
from app.pipeline_helper.schedulermanager import SchedulerManager
from app.pipeline_helper.trainingmanager import TrainingManager
from app.pipeline_helper.testmanager import TestManager
from app.pipeline_helper.dataloader import TransformDataLoader
from app.pipeline_helper.evaluationmanager import EvaluationManager
import os
import torch
from loguru import logger

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

        model = model_manager.get_model()
        
        # Loss Functions
        crop_criterion = torch.nn.CrossEntropyLoss()
        state_criterion = torch.nn.CrossEntropyLoss()
        
        # Optimisers
        optimiser_manager = OptimiserManager(model=model, config=self.config, output_directory=self.output_directory)
        optimiser = optimiser_manager.get_optimiser()
        
        # Learning Rate Scheduler
        scheduler_manager = SchedulerManager(optimiser, config=self.config, output_directory=self.output_directory)
        scheduler = scheduler_manager.get_scheduler()
        
        training_manager = TrainingManager(
            config=self.config,
            output_directory=self.output_directory,
            train_data=train_dataset,
            train_transformers=train_transformers,
            val_transformers=val_transformers,
            model_manager=model_manager,
            crop_criterion=crop_criterion,
            state_criterion=state_criterion,
            optimiser=optimiser,
            scheduler=scheduler
        )
        
        training_manager.start_training()
        
        logger.info(f"Training complete with best validation: {model_manager.model_meta.performance_metrics}")
        
        model_manager.save_model()
        optimiser_manager.save_optimiser()
        scheduler_manager.save_scheduler()
        
    def test(self):
        
        # Datasets
        dataset_manager = DatasetManager(config=self.config, output_directory=self.output_directory, load_existing_structure=False)
        test_dataset = dataset_manager.get_test_dataset()
        
        # Transformers
        transformer_manager = TransformerManager()
        test_transformers = transformer_manager.get_test_transformers()
        
        # Models
        model_manager = ModelManager(
            config=self.config,
            output_directory=self.output_directory,
            data_structure=dataset_manager.structure,
            eval=True
        )
        
        # Loss Functions
        crop_criterion = torch.nn.CrossEntropyLoss()
        state_criterion = torch.nn.CrossEntropyLoss()
        
        test_manager = TestManager(
            config=self.config,
            output_directory=self.output_directory,
            test_data=test_dataset,
            test_transformers=test_transformers,
            model_manager=model_manager,
            crop_criterion=crop_criterion,
            state_criterion=state_criterion,
            data_structure=dataset_manager.structure
        )
        
        test_manager.start_testing()
        
        logger.info(f"Testing complete")
        
        model_manager.save_model()
        
    def evaluate(self):
        
        # Models
        model_manager = ModelManager(
            config=self.config,
            output_directory=self.output_directory,
            eval=True
        )
        
        # Evaluate Training & Validation
        evaluation_manager = EvaluationManager(
            config=self.config,
            output_directory=self.output_directory,
            model_manager=model_manager
        )
        
        # Evaluate Testing
        

    def predict(self):
        pass