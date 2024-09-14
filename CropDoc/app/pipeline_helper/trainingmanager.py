

class TrainingManager():
    def __init__(self,
                 config,
                 output_directory,
                 model,
                 optimiser,
                 scheduler,
                 crop_criterion,
                 state_criterion,
                 train_dataset,
                 train_transformers,
                 val_transformers):
        self.config = config,
        self.output_directory = output_directory
        self.model = model
        self.optimiser = optimiser
        self.scheduler = scheduler
        self.crop_criterion = crop_criterion
        self.state_criterion = state_criterion
        self.train_dataset = train_dataset
        self.train_transformers = train_transformers
        self.val_transformers = val_transformers
        
        self.epochs = self.config['train']['epochs']
        
    def start_training(self)