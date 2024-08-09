import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
from app.service.data import DatasetManager, TransformerManager
from pprint import pprint
from loguru import logger
from sklearn.model_selection import KFold
from tqdm import trange
import shutil
import os
import torch
import matplotlib.pyplot as plt
import warnings
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import json


warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.parallel.parallel_apply")

 

def run( config: dict, dataset_path: str = None):
    """ Run the ResNet50v2 pipeline

    Args:
        config (dict): Configuration dictionary
        dataset_path (str): Path to the dataset (Optional)
        
    """
    logger.debug("Running ResNet50v2 Pipeline")
    
    # Parse config
    try:
        model_config = config['model']
        pipeline_config = config['pipeline']
    except Exception as e:
        logger.error(f"Error parsing config: {e}")
        raise e
    
    # Initialise transformers that will transform the images
    try:
        transformer_manager = TransformerManager(model_config['data']['transformers'])
    except Exception as e:
        logger.error(f"Error initialising transformers: {e}")
        raise e
    
    # Check whether dataset path was provided in config, if not use parameter
    try:
        if pipeline_config['dataset_path'] is not None:
            dataset_path = pipeline_config['dataset_path']
            logger.warning('Overriding dataset path with the one specified in the pipeline config')
        
        if dataset_path is None:
            raise ValueError('Dataset path not specified')
    except Exception as e:
        logger.error(f"Error parsing dataset path: {e}")
        raise e
    
    # Initialise dataset manager which will load the dataset
    try:
        dataset_manager = DatasetManager(dataset_path, transform=transformer_manager.transformers, reduce_dataset=model_config['data']['usage'])
    except Exception as e:
        logger.error(f"Error initialising dataset manager: {e}")
        raise e
    
    # Initialise pipeline
    try:
        pipeline = ResNet50v2Pipeline(model_config, dataset_manager, output_dir=pipeline_config['output_dir'])
    except Exception as e:
        logger.error(f"Error initialising pipeline: {e}")
        raise e
        
    # Train the model
    try:
        pipeline.train()
    except Exception as e:
        logger.error(f"Error training pipeline: {e}")
        raise e
    
    # Save the trained model
    try:
        pipeline.save_trained_model()
    except Exception as e:
        logger.error(f"Error saving trained model: {e}")
        raise e
    
    # Plot training and validation error
    try:
        pipeline.plot_training_validation_error()
    except Exception as e:
        logger.error(f"Error plotting training and validation error: {e}")
        raise e
    
    # Test the model
    try:
        results = pipeline.test()
        results = pipeline.test_and_evaluate()
    except Exception as e:
        logger.error(f"Error testing pipeline: {e}")
        raise e
    
    try:
        pipeline.save_evaluation_results(results)
    except Exception as e:
        logger.error(f"Error saving evaluation results: {e}")
        raise e
    
    # Save the tested model
    try:
        pipeline.save_tested_model()
    except Exception as e:
        logger.error(f"Error saving tested model: {e}")
        raise e
    
    logger.info("Pipeline completed successfully")


class ResNet50v2Pipeline:
    """ Pipeline for training and testing ResNet50v2 model
    """
    def __init__(self, config, dataset: DatasetManager, output_dir: str):
        logger.debug("Initialising ResNet50v2 Pipeline")
        
        # Config used to configure the pipeline
        self.config = config
        self.dataset = dataset
        self.output_dir = output_dir
        
        # Pipeline main objects
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._create_model()
        
        # Loss functions
        self.criterion_crop, self.criterion_state = self._get_loss_functions()
        
        # Optimisers
        self.optimizer = self._get_optimizer()
        
        # Attribute for progress bar
        self.terminal_width = self._get_terminal_width()
        
        # Plot Metrics
        self.batch_train_crop_loss = []
        self.batch_train_state_loss = []
        self.batch_val_crop_loss = []
        self.batch_val_state_loss = []
        
        self.fold_train_crop_loss = []
        self.fold_train_state_loss = []
        self.fold_val_crop_loss = []
        self.fold_val_state_loss = []
        
        self.epoch_train_crop_loss = []
        self.epoch_train_state_loss = []
        self.epoch_val_crop_loss = []
        self.epoch_val_state_loss = []
        
        logger.info("ResNet50v2 Pipeline Initialised")

    def _create_model(self) -> nn.DataParallel:
        """Create ResNet50v2 model

        Returns:
            nn.DataParallel: ResNet50v2 model
        """
        logger.debug("Creating ResNet50v2 Model")
        try:
            # Load ResNet50 model with pre-trained weights
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            
            # Replace the final layer with Identity to make it multi-output classifier
            num_ftrs = model.fc.in_features
            model.fc = nn.Identity()
            model.fc_crop = nn.Linear(num_ftrs, len(self.dataset.unique_crops))
            model.fc_state = nn.Linear(num_ftrs, len(self.dataset.unique_states))
            
            # Make the model parallel if multiple GPUs are available
            model = nn.DataParallel(model)
            
            # Move model to device
            model = model.to(self.device)
            
            logger.info("ResNet50v2 Model Created")
            return model
        
        except Exception as e:
            logger.error(f"Error creating model: {e}")
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

    def _get_optimizer(self):
        """ Get optimiser from config
        """
        optimizer_config = self.config['training']['optimiser']
        optimizer_type = optimizer_config['type']
        optimizer_params = optimizer_config['params']
        
        torch_optimisers = {
            'Adadelta': torch.optim.Adadelta,
            'Adagrad': torch.optim.Adagrad,
            'Adam': torch.optim.Adam,
            'AdamW': torch.optim.AdamW,
            'SparseAdam': torch.optim.SparseAdam,
            'Adamax': torch.optim.Adamax,
            'ASGD': torch.optim.ASGD,
            'LBFGS': torch.optim.LBFGS,
            'NAdam': torch.optim.NAdam,
            'RAdam': torch.optim.RAdam,
            'RMSprop': torch.optim.RMSprop,
            'Rprop': torch.optim.Rprop,
            'SGD': torch.optim.SGD
        }
        
        if optimizer_type not in torch_optimisers.keys():
            logger.error(f"Unsupported optimizer: {optimizer_type}")
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        logger.info(f"Optimizer obtained: {optimizer_type}")
        return torch_optimisers[optimizer_type](self.model.parameters(), **(optimizer_params or {}))

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
    
    def _get_data_loaders(self, train_indices, val_indices):
        train_dataset = torch.utils.data.Subset(self.dataset.train_samples, train_indices)
        val_dataset = torch.utils.data.Subset(self.dataset.train_samples, val_indices)
        
        train_loader = self.create_data_loader(train_dataset, shuffle=True)
        val_loader = self.create_data_loader(val_dataset, shuffle=False)
        
        return train_loader, val_loader

    def _get_terminal_width(self):
        return int(shutil.get_terminal_size().columns * 0.9)
    
    def _set_pbar_postfix(self, pbar, tlc, vlc, tls, vls, tac, vac, tas, vas):
        pbar.set_postfix({
            'TLC(VLC)': f"{tlc:.2f}({vlc:.2f})",
            'TLS(VLS)': f"{tls:.2f}({vls:.2f})",
            'TAC(VAC)': f"{tac * 100:.1f}%({vac * 100:.1f}%)",
            'TAS(VAS)': f"{tas * 100:.1f}%({vas * 100:.1f}%)"
        })
    
    def train(self):
        logger.debug("Training ResNet50v2 Model")
        k_folds = self.config['training']['cv_folds']
        if k_folds > 1:
            kf = KFold(n_splits=self.config['training']['cv_folds'], shuffle=True, random_state=42)
        else:
            full_range = range(len(self.dataset.train_samples))
            kf = [(full_range, full_range)]
        
        # Print helpful information for progress bar
        print(
            "-------------------------------------------------------\n"
            "Progress Bar Postfix Key:\n"
            "T: Train, V: Validated\n"
            "L: Loss, A: Accuracy\n"
            "C: Crop, S: State\n"
            "i.e. TLC: Train Loss Crop, VAS: Val Accuracy State\n"
            "Values in brackets are for validation set\n"
            "-------------------------------------------------------\n"
        )
        
        epoch_pbar = trange(int(self.config['training']['epochs']), desc="Epochs", ncols=max(120, self.terminal_width))
        
        for epoch in epoch_pbar:
            epoch_metrics = self.train_epoch(kf)
            
            # Add metrics to plot later
            self.epoch_train_crop_loss.append(epoch_metrics['train_loss_crop'])
            self.epoch_train_state_loss.append(epoch_metrics['train_loss_state'])
            self.epoch_val_crop_loss.append(epoch_metrics['val_loss_crop'])
            self.epoch_val_state_loss.append(epoch_metrics['val_loss_state'])
            
            # Update epoch progress bar with metrics
            self._set_pbar_postfix(
                epoch_pbar,
                tlc=epoch_metrics['train_loss_crop'], vlc=epoch_metrics['val_loss_crop'],
                tls=epoch_metrics['train_loss_state'], vls=epoch_metrics['val_loss_state'],
                tac=epoch_metrics['train_acc_crop'], vac=epoch_metrics['val_acc_crop'],
                tas=epoch_metrics['train_acc_state'], vas=epoch_metrics['val_acc_state']
            )
        
        logger.info("Training Complete")
        
        # Export epoch metrics to txt file
        with open('epoch_metrics.txt', 'w') as f:
            f.write('Epoch Train Crop Loss\n')
            f.write(str(self.epoch_train_crop_loss))
            f.write('\n\nEpoch Train State Loss\n')
            f.write(str(self.epoch_train_state_loss))
            f.write('\n\nEpoch Val Crop Loss\n')
            f.write(str(self.epoch_val_crop_loss))
            f.write('\n\nEpoch Val State Loss\n')
            f.write(str(self.epoch_val_state_loss))

    def train_epoch(self, kf):
        
        epoch_metrics = {
            'train_loss_crop': 0, 'train_loss_state': 0,
            'train_acc_crop': 0, 'train_acc_state': 0,
            'val_loss_crop': 0, 'val_loss_state': 0,
            'val_acc_crop': 0, 'val_acc_state': 0
        }
        
        fold_pbar = tqdm(enumerate(kf.split(self.dataset.train_samples)),
                         total=self.config['training']['cv_folds'],
                         desc="Folds", leave=False, ncols=max(120, self.terminal_width))
        
        for fold_idx, (train_idx, val_idx) in fold_pbar:
            
            train_loader, val_loader = self._get_data_loaders(train_idx, val_idx)
            
            fold_train_metrics = self.train_fold(fold_idx, train_loader)
            fold_val_metrics = self.validate_fold(fold_idx, val_loader)
            
            # Add metrics to plot later
            self.fold_train_crop_loss.append(fold_train_metrics['train_loss_crop'])
            self.fold_train_state_loss.append(fold_train_metrics['train_loss_state'])
            self.fold_val_crop_loss.append(fold_val_metrics['val_loss_crop'])
            self.fold_val_state_loss.append(fold_val_metrics['val_loss_state'])
            
            for key in epoch_metrics:
                try:
                    if 'train' in key:
                        epoch_metrics[key] += fold_train_metrics[key]
                    elif 'val':
                        epoch_metrics[key] += fold_val_metrics[key]
                    else:
                        raise ValueError(f"Unknown key: {key}")
                except Exception as e:
                    logger.error(
                        f"Error updating epoch metrics:\n"
                        f"Key: {key}\n"
                        f"Epoch Metrics: {epoch_metrics}\n"
                        f"Fold Train Metrics: {fold_train_metrics}\n"
                        f"Fold Val Metrics: {fold_val_metrics}\n"
                        f"Error: {e}"
                    )
                    raise e
            try:
                self._set_pbar_postfix(
                    fold_pbar,
                    tlc=fold_train_metrics['train_loss_crop'], vlc=fold_val_metrics['val_loss_crop'],
                    tls=fold_train_metrics['train_loss_state'], vls=fold_val_metrics['val_loss_state'],
                    tac=fold_train_metrics['train_acc_crop'], vac=fold_val_metrics['val_acc_crop'],
                    tas=fold_train_metrics['train_acc_state'], vas=fold_val_metrics['val_acc_state']
                )
            except Exception as e:
                logger.error(
                    "Error updating epoch metrics:\n"
                    f"Epoch Metrics: {epoch_metrics}\n"
                    f"Fold Train Metrics: {fold_train_metrics}\n"
                    f"Fold Val Metrics: {fold_val_metrics}\n"
                    f"Error: {e}"
                )
                raise e
        
        num_folds = self.config['training']['cv_folds']
        for key in epoch_metrics:
            epoch_metrics[key] /= num_folds
        
        return epoch_metrics
        
    def train_fold(self, fold_idx, train_loader):
        
        self.model.train()
        
        train_loss_crop = 0
        train_loss_state = 0
        train_correct_crop = 0
        train_correct_state = 0
        train_total = 0
        
        batch_pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                          desc="Train Batches", leave=False, ncols=max(120, self.terminal_width))
        
        for batch_idx, batch in batch_pbar:
            crop_label_idx = batch['crop_idx']
            img_paths = batch['img_path']
            splits = batch['split']
            state_label_idx = batch['state_idx']
            
            # Load batch of images
            images = []
            for path, split in zip(img_paths, splits):
                images.append(self.dataset.load_image_from_path(path, split))
            
            images_tensor = torch.stack(images, dim=0)
            
            batch_metrics = self.train_batch(batch_idx, images_tensor, crop_label_idx, state_label_idx)
            
            train_loss_crop += batch_metrics['train_loss_crop']
            train_loss_state += batch_metrics['train_loss_state']
            train_correct_crop += batch_metrics['train_correct_crop']
            train_correct_state += batch_metrics['train_correct_state']
            train_total += batch_metrics['train_total']
            
            self._set_pbar_postfix(
                batch_pbar,
                tlc=batch_metrics['train_loss_crop'], vlc=0,
                tls=batch_metrics['train_loss_state'], vls=0,
                tac=batch_metrics['train_correct_crop'] / batch_metrics['train_total'], vac=0,
                tas=batch_metrics['train_correct_state'] / batch_metrics['train_total'], vas=0
            )

            # Add metrics to plot later
            self.batch_train_crop_loss.append(batch_metrics['train_loss_crop'])
            self.batch_train_state_loss.append(batch_metrics['train_loss_state'])
            
        return {
            'train_loss_crop': train_loss_crop / len(train_loader),
            'train_loss_state': train_loss_state / len(train_loader),
            'train_acc_crop': train_correct_crop / train_total,
            'train_acc_state': train_correct_state / train_total,
            'train_total': train_total
        }
        
    def train_batch(self, batch_idx, inputs, crop_labels, state_labels):
        # Convert inputs to PyTorch tensor if it's not already
        
        inputs = inputs.clone().detach().requires_grad_(True)
        crop_labels = crop_labels.clone().detach()
        state_labels = state_labels.clone().detach()
        
        # Move data to device
        inputs = inputs.to(self.device)
        crop_labels = crop_labels.to(self.device)
        state_labels = state_labels.to(self.device)

        # Zero the gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        model_outputs = self.model(inputs)
        
        crop_outputs = model_outputs[:, :len(self.dataset.unique_crops)]
        state_outputs = model_outputs[:, len(self.dataset.unique_states):]

        # Calculate loss
        loss_crop = self.criterion_crop(crop_outputs, crop_labels)
        loss_state = self.criterion_state(state_outputs, state_labels)
        
        # Backward pass
        loss = loss_crop + loss_state  # Computationally more efficient while still allowing model to learn both tasks
        loss.backward()
        
        # Optimise
        self.optimizer.step()
        
        # Compute statistics
        _, predicted_crop = torch.max(crop_outputs, 1)
        _, predicted_state = torch.max(state_outputs, 1)
        correct_crop = (predicted_crop == crop_labels).sum().item()
        correct_state = (predicted_state == state_labels).sum().item()
        total = crop_labels.size(0)
        
        return {
            'train_loss_crop': loss_crop.item(),
            'train_loss_state': loss_state.item(),
            'train_correct_crop': correct_crop,
            'train_correct_state': correct_state,
            'train_total': total
        }
        
    def validate_fold(self, fold_idx, val_loader):
        self.model.eval()
        val_loss_crop = 0
        val_loss_state = 0
        val_correct_crop = 0
        val_correct_state = 0
        val_total = 0
        
        batch_pbar = tqdm(enumerate(val_loader), total=len(val_loader),
                          desc="Validate Batches", leave=False, ncols=max(120, self.terminal_width))
        
        for batch_idx, batch in batch_pbar:
            crop_label_idx = batch['crop_idx']
            img_paths = batch['img_path']
            splits = batch['split']
            state_label_idx = batch['state_idx']
            
            # Load batch of images
            images = []
            for path, split in zip(img_paths, splits):
                images.append(self.dataset.load_image_from_path(path, split))
            
            images_tensor = torch.stack(images, dim=0)
            
            batch_metrics = self.validate_batch(batch_idx, images_tensor, crop_label_idx, state_label_idx)
            
            val_loss_crop += batch_metrics['val_loss_crop']
            val_loss_state += batch_metrics['val_loss_state']
            val_correct_crop += batch_metrics['val_correct_crop']
            val_correct_state += batch_metrics['val_correct_state']
            val_total += batch_metrics['val_total']
            
            self._set_pbar_postfix(
                batch_pbar,
                tlc=0, vlc=batch_metrics['val_loss_crop'],
                tls=0, vls=batch_metrics['val_loss_state'],
                tac=0, vac=batch_metrics['val_correct_crop'] / batch_metrics['val_total'],
                tas=0, vas=batch_metrics['val_correct_state'] / batch_metrics['val_total']
            )
            
        return {
            'val_loss_crop': val_loss_crop / len(val_loader),
            'val_loss_state': val_loss_state / len(val_loader),
            'val_acc_crop': val_correct_crop / val_total,
            'val_acc_state': val_correct_state / val_total,
            'val_total': val_total
        }
            
    def validate_batch(self, batch_idx, inputs, crop_labels, state_labels):
        # Convert inputs to PyTorch tensor if it's not already
        
        inputs = inputs.clone().detach().requires_grad_(False)
        crop_labels = crop_labels.clone().detach()
        state_labels = state_labels.clone().detach()
        
        # Move data to device
        inputs = inputs.to(self.device)
        crop_labels = crop_labels.to(self.device)
        state_labels = state_labels.to(self.device)

        # Forward pass
        model_outputs = self.model(inputs)
        
        crop_outputs = model_outputs[:, :len(self.dataset.unique_crops)]
        state_outputs = model_outputs[:, len(self.dataset.unique_states):]

        # Calculate loss
        loss_crop = self.criterion_crop(crop_outputs, crop_labels)
        loss_state = self.criterion_state(state_outputs, state_labels)
        
        # Compute statistics
        _, predicted_crop = torch.max(crop_outputs, 1)
        _, predicted_state = torch.max(state_outputs, 1)
        correct_crop = (predicted_crop == crop_labels).sum().item()
        correct_state = (predicted_state == state_labels).sum().item()
        total = crop_labels.size(0)
        
        return {
            'val_loss_crop': loss_crop.item(),
            'val_loss_state': loss_state.item(),
            'val_correct_crop': correct_crop,
            'val_correct_state': correct_state,
            'val_total': total
        }
    
    def test(self):
        logger.debug("Testing ResNet50v2 Model")
        
        # Set model in to evaluation mode
        self.model.eval()
        
        # Initialise test metrics
        test_loss_crop = 0
        test_loss_state = 0
        test_correct_crop = 0
        test_correct_state = 0
        test_total = 0

        # Create test data loader
        test_loader = self.create_data_loader(self.dataset.test_samples, shuffle=False)
        
        # With torch.no_grad() disables gradient calculation, saves memory
        with torch.no_grad():
            
            # Iterate over test data loader
            for batch in test_loader:
                
                # Extract batch data
                crop_label_idx = batch['crop_idx']
                img_paths = batch['img_path']
                splits = batch['split']
                state_label_idx = batch['state_idx']

                # Load images from paths
                images = []
                for path, split in zip(img_paths, splits):
                    images.append(self.dataset.load_image_from_path(path, split))
                images_tensor = torch.stack(images, dim=0).to(self.device)
                
                # Move labels to device
                crop_labels = crop_label_idx.to(self.device)
                state_labels = state_label_idx.to(self.device)

                # Forward pass and obtain outputs
                outputs = self.model(images_tensor)
                crop_outputs = outputs[:, :len(self.dataset.unique_crops)]
                state_outputs = outputs[:, len(self.dataset.unique_crops):]

                # Calculate loss
                loss_crop = self.criterion_crop(crop_outputs, crop_labels)
                loss_state = self.criterion_state(state_outputs, state_labels)

                # Update test metrics
                test_loss_crop += loss_crop.item()
                test_loss_state += loss_state.item()

                # Compute statistics
                _, predicted_crop = torch.max(crop_outputs, 1)
                _, predicted_state = torch.max(state_outputs, 1)
                test_correct_crop += (predicted_crop == crop_labels).sum().item()
                test_correct_state += (predicted_state == state_labels).sum().item()
                test_total += crop_labels.size(0)

        test_loss_crop /= len(test_loader)
        test_loss_state /= len(test_loader)
        test_acc_crop = test_correct_crop / test_total
        test_acc_state = test_correct_state / test_total

        logger.info(f"Test Loss Crop: {test_loss_crop:.4f}")
        logger.info(f"Test Loss State: {test_loss_state:.4f}")
        logger.info(f"Test Accuracy Crop: {test_acc_crop:.4f}")
        logger.info(f"Test Accuracy State: {test_acc_state:.4f}")

        return {
            'test_loss_crop': test_loss_crop,
            'test_loss_state': test_loss_state,
            'test_acc_crop': test_acc_crop,
            'test_acc_state': test_acc_state
        }

    def save_trained_model(self):
        model_dir = os.path.join(self.output_dir, 'trained_model')
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'resnet50v2_trained.pth')
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Trained model saved to {model_path}")

    def save_tested_model(self):
        model_dir = os.path.join(self.output_dir, 'tested_model')
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'resnet50v2_tested.pth')
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Tested model saved to {model_path}")

    def test_and_evaluate(self):
        logger.debug("Testing and Evaluating ResNet50v2 Model")
        
        # Set model in to evaluation mode
        self.model.eval()
        
        # Initialise test metrics
        test_loss_crop = 0
        test_loss_state = 0
        test_correct_crop = 0
        test_correct_state = 0
        test_total = 0
        
        # Initialise lists to store true and predicted labels
        all_true_labels = []
        all_predicted_labels = []

        # Create test data loader
        test_loader = self.create_data_loader(self.dataset.test_samples, shuffle=False)

        # With torch.no_grad() disables gradient calculation, saves memory
        with torch.no_grad():
            
            # Iterate over test data loader
            for batch in test_loader:
                
                # Extract batch data
                crop_label_idx = batch['crop_idx']
                img_paths = batch['img_path']
                splits = batch['split']
                state_label_idx = batch['state_idx']

                # Load images from paths
                images = []
                for path, split in zip(img_paths, splits):
                    images.append(self.dataset.load_image_from_path(path, split))
                images_tensor = torch.stack(images, dim=0).to(self.device)
                
                # Move labels to device
                crop_labels = crop_label_idx.to(self.device)
                state_labels = state_label_idx.to(self.device)

                # Forward pass and obtain outputs
                outputs = self.model(images_tensor)
                crop_outputs = outputs[:, :len(self.dataset.unique_crops)]
                state_outputs = outputs[:, len(self.dataset.unique_crops):]

                # Calculate loss
                loss_crop = self.criterion_crop(crop_outputs, crop_labels)
                loss_state = self.criterion_state(state_outputs, state_labels)

                # Update test metrics
                test_loss_crop += loss_crop.item()
                test_loss_state += loss_state.item()

                # Compute statistics
                _, predicted_crop = torch.max(crop_outputs, 1)
                _, predicted_state = torch.max(state_outputs, 1)
                test_correct_crop += (predicted_crop == crop_labels).sum().item()
                test_correct_state += (predicted_state == state_labels).sum().item()
                test_total += crop_labels.size(0)

                # Append true and predicted labels
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

        # Compute evaluation metrics
        crop_precision = precision_score(all_true_labels[:len(all_true_labels) // 2], all_predicted_labels[:len(all_predicted_labels) // 2], average='macro')
        crop_recall = recall_score(all_true_labels[:len(all_true_labels) // 2], all_predicted_labels[:len(all_predicted_labels) // 2], average='macro')
        crop_f1 = f1_score(all_true_labels[:len(all_true_labels) // 2], all_predicted_labels[:len(all_predicted_labels) // 2], average='macro')

        state_precision = precision_score(all_true_labels[len(all_true_labels) // 2:], all_predicted_labels[len(all_predicted_labels) // 2:], average='macro')
        state_recall = recall_score(all_true_labels[len(all_true_labels) // 2:], all_predicted_labels[len(all_predicted_labels) // 2:], average='macro')
        state_f1 = f1_score(all_true_labels[len(all_true_labels) // 2:], all_predicted_labels[len(all_predicted_labels) // 2:], average='macro')

        # Compute confusion matrix
        crop_cm = confusion_matrix(all_true_labels[:len(all_true_labels) // 2], all_predicted_labels[:len(all_predicted_labels) // 2])
        state_cm = confusion_matrix(all_true_labels[len(all_true_labels) // 2:], all_predicted_labels[len(all_predicted_labels) // 2:])

        print("--------------------------------------------------------------")
        print('Confusion Matrices\n')
        print("Crop Confusion Matrix:")
        print(crop_cm)
        print("State Confusion Matrix:")
        print(state_cm)
        print("--------------------------------------------------------------")
        
        # Save evaluation results
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
    
    def plot_training_validation_error(self):
        plot_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)

        # Plot epoch losses
        plt.figure(figsize=(10, 5))
        plt.plot(self.epoch_train_crop_loss, label='Train Crop Loss')
        plt.plot(self.epoch_train_state_loss, label='Train State Loss')
        plt.plot(self.epoch_val_crop_loss, label='Val Crop Loss')
        plt.plot(self.epoch_val_state_loss, label='Val State Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss per Epoch')
        plt.legend()
        plt.savefig(os.path.join(plot_dir, 'epoch_losses.png'))
        plt.close()

        # Plot fold losses
        plt.figure(figsize=(10, 5))
        plt.plot(self.fold_train_crop_loss, label='Train Crop Loss')
        plt.plot(self.fold_train_state_loss, label='Train State Loss')
        plt.plot(self.fold_val_crop_loss, label='Val Crop Loss')
        plt.plot(self.fold_val_state_loss, label='Val State Loss')
        plt.xlabel('Fold')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss per Fold')
        plt.legend()
        plt.savefig(os.path.join(plot_dir, 'fold_losses.png'))
        plt.close()

        # Plot batch losses
        plt.figure(figsize=(10, 5))
        plt.plot(self.batch_train_crop_loss, label='Train Crop Loss')
        plt.plot(self.batch_train_state_loss, label='Train State Loss')
        plt.plot(self.batch_val_crop_loss, label='Val Crop Loss')
        plt.plot(self.batch_val_state_loss, label='Val State Loss')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss per Batch')
        plt.legend()
        plt.savefig(os.path.join(plot_dir, 'batch_losses.png'))
        plt.close()

        logger.info(f"Training and validation error plots saved in {plot_dir}")
    
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

    def plot_roc_curve(self, crop_fpr, crop_tpr, crop_roc_auc, state_fpr, state_tpr, state_roc_auc):
        plot_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        # Plot crop ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(crop_fpr, crop_tpr, color='darkorange', lw=2, label=f'Crop ROC curve (AUC = {crop_roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Crop ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(plot_dir, 'crop_roc_curve.png'))
        plt.close()
        
        # Plot state ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(state_fpr, state_tpr, color='darkorange', lw=2, label=f'State ROC curve (AUC = {state_roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('State ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(plot_dir, 'state_roc_curve.png'))
        plt.close()
    
    def save_test_results(self, test_metrics):
        test_results_dir = os.path.join(self.output_dir, 'test_results')
        os.makedirs(test_results_dir, exist_ok=True)

        test_results_path = os.path.join(test_results_dir, 'test_results.json')
        with open(test_results_path, 'w') as f:
            json.dump(test_metrics, f, indent=4)

        logger.info(f"Test results saved to {test_results_path}")