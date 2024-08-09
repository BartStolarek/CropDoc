import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
from app.service.data import DatasetManager, TransformerManager
from loguru import logger
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import json


def run_test_pipeline(config: dict, dataset_path: str, model_path: str):
    logger.debug("Running ResNet50v2 Test Pipeline")
    
    try:
        model_config = config['model']
        pipeline_config = config['pipeline']
    except Exception as e:
        logger.error(f"Error parsing config: {e}")
        raise e
    
    try:
        transformer_manager = TransformerManager(model_config['data']['transformers'])
    except Exception as e:
        logger.error(f"Error initialising transformers: {e}")
        raise e
    
    try:
        dataset_manager = DatasetManager(dataset_path, transform=transformer_manager.transformers, reduce_dataset=model_config['data']['usage'])
    except Exception as e:
        logger.error(f"Error initialising dataset manager: {e}")
        raise e
    
    try:
        pipeline = ResNet50v2TestPipeline(model_config, dataset_manager, output_dir=pipeline_config['output_dir'], model_path=model_path)
    except Exception as e:
        logger.error(f"Error initialising pipeline: {e}")
        raise e
        
    try:
        results = pipeline.test()
        
    except Exception as e:
        logger.error(f"Error testing pipeline: {e}")
        raise e
    
    logger.info("Test Pipeline completed successfully")

class ResNet50v2TestPipeline:
    def __init__(self, config, dataset: DatasetManager, output_dir: str, model_path: str):
        logger.debug("Initialising ResNet50v2 Test Pipeline")
        
        self.config = config
        self.dataset = dataset
        self.output_dir = output_dir
        self.model_path = model_path
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        
        self.criterion_crop, self.criterion_state = self._get_loss_functions()
        
        logger.info("ResNet50v2 Test Pipeline Initialised")

    def _load_model(self) -> nn.DataParallel:
        logger.debug("Loading ResNet50v2 Model")
        try:
            model = models.resnet50()
            num_ftrs = model.fc.in_features
            model.fc = nn.Identity()
            model.fc_crop = nn.Linear(num_ftrs, len(self.dataset.unique_crops))
            model.fc_state = nn.Linear(num_ftrs, len(self.dataset.unique_states))
            
            model = nn.DataParallel(model)
            model.load_state_dict(torch.load(self.model_path))
            model = model.to(self.device)
            
            logger.info(f"ResNet50v2 Model Loaded from {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e

    def _get_loss_functions(self):
        """ Get loss functions from config

        """
        logger.debug("Getting Loss Functions")
    
        try:
            loss_type = self.config['training']['loss_function']['type']
        except KeyError as e:
            logger.error(f"Potentially missing config['training']['loss_function']['type'] from config file in CropDoc/app/config/<your config file> directory: {e}")
            raise ValueError('Missing loss function type in config file')
        except Exception as e:
            logger.error(f"Error parsing loss function: {e}")
            raise e
        try:
            loss_params = self.config['training']['loss_function']['params']
        except KeyError as e:
            logger.info(f'No loss function parameters provided: {e}')
            loss_params = None
        
        nn_loss_functions = {
            'L1Loss': nn.L1Loss,
            'MSELoss': nn.MSELoss,
            'CrossEntropyLoss': nn.CrossEntropyLoss,
            'CTCLoss': nn.CTCLoss,
            'NLLLoss': nn.NLLLoss,
            'PoissonNLLLoss': nn.PoissonNLLLoss,
            'GaussianNLLLoss': nn.GaussianNLLLoss,
            'KLDivLoss': nn.KLDivLoss,
            'BCELoss': nn.BCELoss,
            'BCEWithLogitsLoss': nn.BCEWithLogitsLoss,
            'MarginRankingLoss': nn.MarginRankingLoss,
            'HingeEmbeddingLoss': nn.HingeEmbeddingLoss,
            'MultiLabelMarginLoss': nn.MultiLabelMarginLoss,
            'HuberLoss': nn.HuberLoss,
            'SmoothL1Loss': nn.SmoothL1Loss,
            'SoftMarginLoss': nn.SoftMarginLoss,
            'MultiLabelSoftMarginLoss': nn.MultiLabelSoftMarginLoss,
            'CosineEmbeddingLoss': nn.CosineEmbeddingLoss,
            'MultiMarginLoss': nn.MultiMarginLoss,
            'TripletMarginLoss': nn.TripletMarginLoss,
            'TripletMarginWithDistanceLoss': nn.TripletMarginWithDistanceLoss
        }
        
        if loss_type not in nn_loss_functions.keys():
            logger.error(f"Unsupported loss function: {loss_type}")
            raise ValueError(f"Unsupported loss function: {loss_type}")
        
        logger.info(f"Loss function obtained: {loss_type}")
        return nn_loss_functions[loss_type](**(loss_params or {})), nn_loss_functions[loss_type](**(loss_params or {}))

    def create_data_loader(self, dataset, shuffle=True):
        """ Create a DataLoader from a Dataset
        """
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=shuffle,
            num_workers=self.config['data']['workers'],
            pin_memory=True if torch.cuda.is_available() else False
        )

    def test(self):
        logger.debug("Testing ResNet50v2 Model")
        
        self.model.eval()
        
        test_loss_crop = 0
        test_loss_state = 0
        test_correct_crop = 0
        test_correct_state = 0
        test_total = 0

        all_true_labels = []
        all_predicted_labels = []

        test_loader = self.create_data_loader(self.dataset.test_samples, shuffle=False)

        with torch.no_grad():
            for batch in test_loader:
                crop_label_idx = batch['crop_idx']
                img_paths = batch['img_path']
                splits = batch['split']
                state_label_idx = batch['state_idx']

                images = []
                for path, split in zip(img_paths, splits):
                    images.append(self.dataset.load_image_from_path(path, split))
                images_tensor = torch.stack(images, dim=0).to(self.device)
                
                crop_labels = crop_label_idx.to(self.device)
                state_labels = state_label_idx.to(self.device)

                outputs = self.model(images_tensor)
                crop_outputs = outputs[:, :len(self.dataset.unique_crops)]
                state_outputs = outputs[:, len(self.dataset.unique_crops):]

                loss_crop = self.criterion_crop(crop_outputs, crop_labels)
                loss_state = self.criterion_state(state_outputs, state_labels)

                test_loss_crop += loss_crop.item()
                test_loss_state += loss_state.item()

                _, predicted_crop = torch.max(crop_outputs, 1)
                _, predicted_state = torch.max(state_outputs, 1)
                test_correct_crop += (predicted_crop == crop_labels).sum().item()
                test_correct_state += (predicted_state == state_labels).sum().item()
                test_total += crop_labels.size(0)

                all_true_labels.extend(crop_label_idx.tolist())
                all_true_labels.extend(state_label_idx.tolist())
                all_predicted_labels.extend(predicted_crop.tolist())
                all_predicted_labels.extend(predicted_state.tolist())

        test_loss_crop /= len(test_loader)
        test_loss_state /= len(test_loader)
        test_acc_crop = test_correct_crop / test_total
        test_acc_state = test_correct_state / test_total

        logger.info(f"Test Loss Crop: {test_loss_crop:.4f}")
        logger.info(f"Test Loss State: {test_loss_state:.4f}")
        logger.info(f"Test Accuracy Crop: {test_acc_crop:.4f}")
        logger.info(f"Test Accuracy State: {test_acc_state:.4f}")

        crop_precision = precision_score(all_true_labels[:len(all_true_labels) // 2], all_predicted_labels[:len(all_predicted_labels) // 2], average='macro')
        crop_recall = recall_score(all_true_labels[:len(all_true_labels) // 2], all_predicted_labels[:len(all_predicted_labels) // 2], average='macro')
        crop_f1 = f1_score(all_true_labels[:len(all_true_labels) // 2], all_predicted_labels[:len(all_predicted_labels) // 2], average='macro')

        state_precision = precision_score(all_true_labels[len(all_true_labels) // 2:], all_predicted_labels[len(all_predicted_labels) // 2:], average='macro')
        state_recall = recall_score(all_true_labels[len(all_true_labels) // 2:], all_predicted_labels[len(all_predicted_labels) // 2:], average='macro')
        state_f1 = f1_score(all_true_labels[len(all_true_labels) // 2:], all_predicted_labels[len(all_predicted_labels) // 2:], average='macro')

        crop_cm = confusion_matrix(all_true_labels[:len(all_true_labels) // 2], all_predicted_labels[:len(all_predicted_labels) // 2])
        state_cm = confusion_matrix(all_true_labels[len(all_true_labels) // 2:], all_predicted_labels[len(all_predicted_labels) // 2:])

        eval_results = {
            'test_loss_crop': test_loss_crop,
            'test_loss_state': test_loss_state,
            'test_acc_crop': test_acc_crop,
            'test_acc_state': test_acc_state,
            'crop_precision': crop_precision,
            'crop_recall': crop_recall,
            'crop_f1': crop_f1,
            'state_precision': state_precision,
            'state_recall': state_recall,
            'state_f1': state_f1,
            'crop_confusion_matrix': crop_cm.tolist(),
            'state_confusion_matrix': state_cm.tolist(),
        }

        self.save_evaluation_results(eval_results)
        self.plot_confusion_matrix(crop_cm, state_cm)

        return eval_results

    def save_evaluation_results(self, eval_results):
        eval_results_dir = os.path.join(self.output_dir, 'evaluation_results')
        os.makedirs(eval_results_dir, exist_ok=True)

        eval_results_path = os.path.join(eval_results_dir, 'evaluation_results.json')
        with open(eval_results_path, 'w') as f:
            json.dump(eval_results, f, indent=4)

        logger.info(f"Evaluation results saved to {eval_results_path}")

    def plot_confusion_matrix(self, crop_cm, state_cm):
        plot_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        # Plot crop confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(crop_cm, cmap='Blues')
        plt.title('Crop Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(plot_dir, 'crop_confusion_matrix.png'))
        plt.close()
        
        # Plot state confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(state_cm, cmap='Blues')
        plt.title('State Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(plot_dir, 'state_confusion_matrix.png'))
        plt.close()



