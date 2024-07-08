# Example configuration module for a crop disease classification model

class CropDiseaseClassifierConfig:
    def __init__(self):
        self.model_name = "CropDiseaseClassifier"
        self.model_type = "CNN"
        self.model_architecture = [
            {"type": "Conv2D", "filters": 32, "kernel_size": 3, "activation": "relu"},
            {"type": "MaxPooling2D", "pool_size": 2},
            {"type": "Flatten"},
            {"type": "Dense", "units": 64, "activation": "relu"},
            {"type": "Dense", "units": 10, "activation": "softmax"}
        ]
        
        self.training_epochs = 50
        self.training_batch_size = 32
        self.training_learning_rate = 0.001
        
        self.data_dataset = "CropDiseaseDataset"
        self.data_train_path = "/path/to/train/data"
        self.data_validation_path = "/path/to/validation/data"
        self.data_image_size = (224, 224)
        self.data_augmentation = [
            {"type": "horizontal_flip"},
            {"type": "rotation", "rotation_range": 20},
            {"type": "brightness", "brightness_range": (0.8, 1.2)}
        ]
        
        self.evaluation_metrics = ["accuracy", "precision", "recall"]
        self.evaluation_batch_size = 64
