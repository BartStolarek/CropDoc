
from app.pipeline_helper.dataset import CropCCMTDataset, PlantVillageDataset

class Pipeline:
    def __init__(self, config):
        self.config = config
    
    def train(self):
        print(f"{self.config['datasets']}")
        dataset = CropCCMTDataset(root=self.config['datasets'][0]['dir'])
        dataset2 = PlantVillageDataset(root=self.config['datasets'][1]['dir'], test_split=self.config['datasets'][1]['test_split'])
        # Datasets
        # - Index Mapping
        # - Transformations
        # - Train Dataset & Validation Dataset
        
        
        # Model
        # - Create new or load existing
        # - Ensure head is the right size for loaded dataset/s
        pass
        
    def test(self):
        pass

    def predict(self):
        pass