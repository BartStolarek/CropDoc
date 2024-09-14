from app.pipeline_helper.dataloader import TransformDataLoader
import torch

class TrainingManager():
    def __init__(self,
                 config,
                 output_directory,
                 train_data,
                 train_transformers,
                 val_transformers,
                 model, crop_criterion,
                 state_criterion,
                 optimiser,
                 scheduler):
        self.output_directory = output_directory
        self.config = config
        
        self.epochs = int(config['train']['epochs'])
        self.checkpoint_interval = int(config['train']['checkpoint_interval'])
        self.validation_split = float(config['train']['validation_split'])
        self.batch_size = int(config['train']['batch_size'])
        self.num_workers = int(config['train']['num_workers'])
        self.model = model
        self.crop_criterion = crop_criterion
        self.state_criterion = state_criterion
        self.optimiser = optimiser
        self.scheduler = scheduler
        
        
        self.train_dataset, val_dataset = torch.utils.data.dataset.random_split(
            dataset=train_data,
            lengths=[
                1 - self.validation_split,
                self.validation_split
            ]
        )
        
        self.train_dataloader = TransformDataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            transform=train_transformers
        )
        
        self.val_dataloader = TransformDataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            transform=val_transformers
        )
        
    def start_training(self):
        pass
    
