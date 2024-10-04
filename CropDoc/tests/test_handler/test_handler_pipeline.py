import pytest
from unittest.mock import patch, MagicMock
from app.handler.pipeline import handle_train, handle_predict, handle_test, handle_predict_many
from app.pipeline.pipeline import Pipeline

@pytest.fixture
def mock_config():
    return {
        "pipeline": {
            "model": "ResNet50",
            "version": "3.1"
        },
        "datasets": [
            {
                "name": "CropCCMTDataset",
                "class": "CropCCMTDataset",
                "root": "data/datasets/CCMT Dataset-Augmented",
                "active": True
            },
            {
                "name": "PlantVillageDataset",
                "class": "PlantVillageDataset",
                "root": "data/datasets/plantvillage/color",
                "test_split": 0.2,
                "active": True
            }
        ],
        "train": {
            "epochs": 50,
            "checkpoint_interval": 5,
            "validation_split": 0.2,
            "batch_size": 384,
            "num_workers": 4,
            "learning_rate": 0.005,
            "lr_scheduler": {
                "active": "ReduceLROnPlateau",
                "StepLR": {
                    "step_size": 30,
                    "gamma": 0.3
                },
                "ReduceLROnPlateau": {
                    "mode": "min",
                    "factor": 0.3,
                    "patience": 10,
                    "threshold": 0.001
                }
            },
            "loss_function": {
                "type": "CrossEntropyLoss"
            }
        },
        "test": {
            "batch_size": 384,
            "num_workers": 4
        },
        "predict": {
            "crop": {
                "confidence_threshold": 0.2
            },
            "state": {
                "confidence_threshold": 0.2
            }
        }
    }

@pytest.fixture
def mock_pipeline(mock_config):
    with patch('app.handler.pipeline.Pipeline') as MockPipeline:
        pipeline_instance = MockPipeline.return_value
        pipeline_instance.train = MagicMock()
        pipeline_instance.predict_model = MagicMock()
        pipeline_instance.test = MagicMock()
        pipeline_instance.save_prediction = MagicMock()
        yield pipeline_instance

@pytest.fixture
def mock_load_yaml(mock_config):
    with patch('app.handler.pipeline.load_yaml_file_as_dict') as mock_load:
        mock_load.return_value = mock_config
        yield mock_load

class TestHandleTrain:
    def test_handle_train_success(self, mock_load_yaml, mock_pipeline):
        handle_train("test_config.yaml")
        mock_load_yaml.assert_called_once_with(directory='config', file_name="test_config.yaml")
        mock_pipeline.train.assert_called_once()

class TestHandlePredict:
    def test_handle_predict_success(self, mock_load_yaml, mock_pipeline):
        # mock_pipeline.predict_model.return_value = {"crop": "wheat", "state": "healthy"}
        # result = handle_predict("test_config.yaml", "path/to/image.jpg")
        # mock_load_yaml.assert_called_once_with(directory='config', file_name="test_config.yaml")
        # mock_pipeline.predict_model.assert_called_once_with("path/to/image.jpg")
        # mock_pipeline.save_prediction.assert_called_once_with({"crop": "wheat", "state": "healthy"})
        # assert result == {"crop": "wheat", "state": "healthy"}       
        handle_predict("test_config.yaml", "path/to/image.jpg")
        mock_load_yaml.assert_called_once_with(directory='config', file_name="test_config.yaml")
        mock_pipeline.predict.assert_called_once()
        mock_pipeline.predict.return_value = {"crop": "tomato", "state": "healthy"}
        mock_pipeline.predict.assert_called_once_with("path/to/image.jpg")
        # mock_pipeline.save_prediction.assert_called_once_with({"crop": "wheat", "state": "healthy"})
        # assert result == {"crop": "wheat", "state": "healthy"}

class TestHandleTest:
    def test_handle_test_success(self, mock_load_yaml, mock_pipeline):
        handle_test("test_config.yaml")
        mock_load_yaml.assert_called_once_with(directory='config', file_name="test_config.yaml")
        mock_pipeline.test.assert_called_once()

