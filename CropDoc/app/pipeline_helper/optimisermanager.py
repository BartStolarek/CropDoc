import os
import torch

class OptimiserManager():
    
    def __init__(self, model, config, output_directory):
        self.model = model
        self.config = config
        self.output_directory = output_directory
        self.optimiser_path = os.path.join(output_directory, 'optimiser', 'optimiser.pth')
        
        self.learning_rate = self.config['training']['learning_rate']
        
        if os.path.exists(self.optimiser_path):
            self.optimiser = self._load_optimiser()
        else:
            self.optimiser = self._create_optimiser()
            
    def _create_optimiser(self):
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimiser
    
    def _load_optimiser(self):
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        optimiser.load_state_dict(torch.load(self.optimiser_path))
        return optimiser
    
    def get_optimiser(self):
        return self.optimiser
    
    def save_optimiser(self, optimiser):
        os.makedirs(os.path.join(self.output_directory, 'optimiser'), exist_ok=True)
        torch.save(optimiser.state_dict(), self.optimiser_path)
        
        