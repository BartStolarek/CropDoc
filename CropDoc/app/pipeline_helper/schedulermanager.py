import torch
import os

class SchedulerManager():
    def __init__(self, optimiser, config, output_directory):
        self.optimiser = optimiser
        self.config = config
        self.mode = self.config['train']['lr_scheduler']['mode']
        self.factor = self.config['train']['lr_scheduler']['factor']
        self.patience = self.config['train']['lr_scheduler']['patience']
        self.threshold = self.config['train']['lr_scheduler']['threshold']
        self.output_directory = output_directory
        self.scheduler_path = os.path.join(output_directory, 'scheduler', 'scheduler.pth')
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser,
            mode=self.mode,
            factor=self.factor,
            patience=self.patience,
            threshold=self.threshold)
        
        if os.path.exists(self.scheduler_path):
            self.scheduler = self._load_scheduler()
        else:
            self.scheduler = self._create_scheduler()
            
    def _create_scheduler(self):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser,
            mode=self.mode,
            factor=self.factor,
            patience=self.patience,
            threshold=self.threshold)
    
    def _load_scheduler(self):
        self.scheduler.load_state_dict(torch.load(self.scheduler_path))
        return self.scheduler

    def get_scheduler(self):
        return self.scheduler
    
    def save_scheduler(self, scheduler):
        os.makedirs(os.path.join(self.output_directory, 'scheduler'), exist_ok=True)
        torch.save(scheduler.state_dict(), self.scheduler_path)
