from CropDoc.app.service.data import ResNet50V2DatasetManager, ResNet50V2ConcatDatasetManager

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
        
        
        