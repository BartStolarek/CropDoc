from app.pipeline_helper.dataloader import TransformDataLoader
from app.pipeline_helper.metricstrackers import PerformanceMetrics, ProgressionMetrics, SplitMetrics, Metrics
from app.pipeline_helper.pipelinemanager import PipelineManager
import torch
from loguru import logger
import shutil
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import os
import pandas as pd
from datetime import datetime
import os


class TestManager(PipelineManager):
    def __init__(self,
                 config,
                 output_directory,
                 test_data,
                 test_transformers,
                 model_manager,
                 crop_criterion,
                 state_criterion,
                 data_structure):
        super().__init__()
        self.output_directory = output_directory
        self.config = config
        
        self.batch_size = int(config['test']['batch_size'])
        self.num_workers = int(config['test']['num_workers'])
        self.model_manager = model_manager
        self.model = model_manager.get_model()
        self.crop_criterion = crop_criterion
        self.state_criterion = state_criterion
        
        self.data_structure = data_structure

        self.test_dataset = test_data
        self.test_dataloader = TransformDataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            transform=test_transformers
        )

        logger.info(f"Test Manager initialised with {len(self.test_dataset)} test samples")
        
    def start_testing(self):
        
        self.performance_metrics = self.model_manager.model_meta.performance_metrics
        
        with torch.no_grad():
            test_metrics = self._feed_model(self.test_dataloader, split='test')
            test_metrics.assign_split('test')
            self.performance_metrics.test = test_metrics
            
        logger.info(f"Testing complete with test metrics: {self.performance_metrics.test}")
        
        # Log test results in to test directory
        self._save_test_results()
        
    def _save_test_results(self):
        # Check if test directory exists
        test_directory = os.path.join(self.output_directory, 'test_results')
        
        # Create test directory if it does not exist
        if not os.path.exists(test_directory):
            os.makedirs(test_directory, exist_ok=True)
         
        # Check if class label maps exist and update them   
        crop_map_path = os.path.join(test_directory, 'crop_map.csv')  
        self.update_dataframe(crop_map_path, datetime.now(), self.model_manager.model_meta.crops.tolist())
        
        state_map_path = os.path.join(test_directory, 'state_map.csv')
        self.update_dataframe(state_map_path, datetime.now(), self.model_manager.model_meta.crops.tolist())
            
        # Save Test Results
        self.save_to_csv(
            directory=test_directory,
            model_name_version=self.model_manager.name_version,
            dataset_name=self.test_dataset.name,
            num_classes_crop=len(self.data_structure.crops),
            num_classes_state=len(self.data_structure.states),
            samples=len(self.test_dataset),
            crop_accuracy=self.performance_metrics.test.crop.accuracy,
            crop_loss=self.performance_metrics.test.crop.loss,
            crop_precision=self.performance_metrics.test.crop.precision,
            crop_recall=self.performance_metrics.test.crop.recall,
            crop_f1=self.performance_metrics.test.crop.f1_score,
            state_accuracy=self.performance_metrics.test.state.accuracy,
            state_loss=self.performance_metrics.test.state.loss,
            state_precision=self.performance_metrics.test.state.precision,
            state_recall=self.performance_metrics.test.state.recall,
            state_f1=self.performance_metrics.test.state.f1_score
        )
    
    def save_to_csv(
        self,
        directory: str,
        model_name_version: str,
        dataset_name: str,
        num_classes_crop: int,
        num_classes_state: int,
        samples: int,
        crop_accuracy: float,
        crop_loss: float,
        crop_precision: float,
        crop_recall: float,
        crop_f1: float,
        state_accuracy: float,
        state_loss: float,
        state_precision: float,
        state_recall: float,
        state_f1: float
    ) -> pd.DataFrame:
        """
        Loads or creates a pandas DataFrame from a CSV file and adds a new row with the provided test results.
        Includes a datetime column for when the results were saved.

        Args:
        [args remain the same]

        Returns:
        pd.DataFrame: Updated DataFrame with the new row added.
        """
        csv_path = os.path.join(directory, "test_results.csv")

        flattened_dataset_names = list(self._flatten(dataset_name))
        data_set_string = '_'.join(flattened_dataset_names)
        
        # Check if CSV file exists and load it, or create a new DataFrame
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            df = pd.DataFrame(columns=[
                "DateTime", "Model", "Dataset", "NumClassesCrop", "NumClassesState", "Samples",
                "CropAccuracy", "CropLoss", "CropPrecision", "CropRecall", "CropF1",
                "StateAccuracy", "StateLoss", "StatePrecision", "StateRecall", "StateF1"
            ])

        # Create a new row with the current data
        new_row = pd.DataFrame({
            "DateTime": [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            "Model": [model_name_version],
            "Dataset": [data_set_string],
            "NumClassesCrop": [num_classes_crop],
            "NumClassesState": [num_classes_state],
            "Samples": [samples],
            "CropAccuracy": [crop_accuracy],
            "CropLoss": [crop_loss],
            "CropPrecision": [crop_precision],
            "CropRecall": [crop_recall],
            "CropF1": [crop_f1],
            "StateAccuracy": [state_accuracy],
            "StateLoss": [state_loss],
            "StatePrecision": [state_precision],
            "StateRecall": [state_recall],
            "StateF1": [state_f1]
        })

        # Concatenate the new row to the DataFrame
        df = pd.concat([df, new_row], ignore_index=True)

        # Save the updated DataFrame back to CSV
        df.to_csv(csv_path, index=False)

        return df
    
    def _flatten(self, nested_list):
        for item in nested_list:
            if isinstance(item, list):
                yield from self._flatten(item)
            else:
                yield item
    
    def update_dataframe(self, csv_path, new_datetime: datetime, string_list: list[str]):
        """
        Updates or creates a DataFrame with a new column of strings, using the provided datetime as the column header.
        If a CSV file exists at the given path, it loads the data from there. Otherwise, it creates a new DataFrame.
        The updated DataFrame is saved back to the CSV file.
        
        Args:
        new_datetime (datetime): Datetime to use as the new column header.
        string_list (list[str]): List of strings to add as the new column.
        
        Returns:
        pd.DataFrame: Updated DataFrame with the new column added.
        """
        # Check if CSV file exists and load it, or create a new DataFrame
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, index_col='Index')
        else:
            df = pd.DataFrame({'Index': range(1, len(string_list) + 1)})
            df.set_index('Index', inplace=True)
        
        # Convert datetime to string format for column name
        datetime_str = new_datetime.strftime('%Y-%m-%d %H:%M:%S')
        
        # Add new column with the datetime as header and string list as values
        df[datetime_str] = string_list
        
        # Save the updated DataFrame back to CSV
        df.to_csv(csv_path)
        