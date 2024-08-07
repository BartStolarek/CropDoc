import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
import os
from app.service.data import DatasetManager, TransformerManager
from torch.utils.data import DataLoader  # PyTorch data utilities
from pprint import pprint
from loguru import logger
from sklearn.model_selection import KFold
import traceback
from tqdm import trange


def run(dataset_path: str, config: dict):
    pprint(config)
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
        if pipeline_config['dataset_path'] is not None:
            dataset_path = pipeline_config['dataset_path']
            logger.warning('Overriding dataset path with the one specified in the pipeline config')
        
        if dataset_path is None:
            raise ValueError('Dataset path not specified')
    except Exception as e:
        logger.error(f"Error parsing dataset path: {e}")
        raise e
    
    try:
        dataset_manager = DatasetManager(dataset_path, transform=transformer_manager.transformers)
    except Exception as e:
        logger.error(f"Error initialising dataset manager: {e}")
        raise e
    try:
        pipeline = ResNet50v2Pipeline(model_config, dataset_manager)
    except Exception as e:
        logger.error(f"Error initialising pipeline: {e}")
        raise e
        
    try:
        pipeline.train()
    except Exception as e:
        logger.error(f"Error training pipeline: {e}")
        raise e
    
    try:
        pipeline.test()
    except Exception as e:
        logger.error(f"Error testing pipeline: {e}")
        raise e

    try:
        pipeline.save_model()
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise e
    
    logger.info("Pipeline completed successfully")


class ResNet50v2Pipeline:
    def __init__(self, config, dataset: DatasetManager):
        logger.debug("Initialising ResNet50v2 Pipeline")
        self.config = config
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = self._create_model()
        self.criterion_crop, self.criterion_state = self._get_loss_functions()
        self.optimizer = self._get_optimizer()
        logger.info("ResNet50v2 Pipeline Initialised")

    def _create_model(self):
        logger.debug("Creating ResNet50v2 Model")
        try:
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            num_ftrs = model.fc.in_features
            model.fc = nn.Identity()
            model.fc_crop = nn.Linear(num_ftrs, len(self.dataset.unique_crops))
            model.fc_state = nn.Linear(num_ftrs, len(self.dataset.unique_states))
            logger.info("ResNet50v2 Model Created")
            return model.to(self.device)
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            raise e

    def _get_loss_functions(self):
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

    def _get_data_loaders(self, train_indices, val_indices):
        
        # Create data samplers which are used to get the data for the training and validation sets
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
        
        # Create data loaders which are used to load the data in batches
        train_loader = torch.utils.data.DataLoader(
            self.dataset.train_samples,
            batch_size=self.config['training']['batch_size'],
            sampler=train_sampler,
            num_workers=self.config['training'].get('num_workers', 0),
            pin_memory=True if torch.cuda.is_available() else False
        )
        val_loader = torch.utils.data.DataLoader(
            self.dataset.train_samples,
            batch_size=self.config['training']['batch_size'],
            sampler=val_sampler,
            num_workers=self.config['training'].get('num_workers', 0),
            pin_memory=True if torch.cuda.is_available() else False
        )
    
        return train_loader, val_loader

    def train(self):
        logger.debug("Training ResNet50v2 Model")
        kf = KFold(n_splits=self.config['training']['cv_folds'], shuffle=True, random_state=42)
        
        epoch_pbar = trange(self.config['training']['epochs'], desc="Epochs")
        for epoch in epoch_pbar:
            epoch_metrics = self.train_epoch(kf)
            
            # Update epoch progress bar with metrics
            epoch_pbar.set_postfix({
                'Train Loss Crop': f"{epoch_metrics['train_loss_crop']:.4f}",
                'Train Loss State': f"{epoch_metrics['train_loss_state']:.4f}",
                'Train Acc Crop': f"{epoch_metrics['train_acc_crop']:.4f}",
                'Train Acc State': f"{epoch_metrics['train_acc_state']:.4f}",
                'Val Loss Crop': f"{epoch_metrics['val_loss_crop']:.4f}",
                'Val Loss State': f"{epoch_metrics['val_loss_state']:.4f}",
                'Val Acc Crop': f"{epoch_metrics['val_acc_crop']:.4f}",
                'Val Acc State': f"{epoch_metrics['val_acc_state']:.4f}"
            })
        
        logger.info("Training Complete")

    def train_epoch(self, kf):
        
        epoch_metrics = {
            'train_loss_crop': 0, 'train_loss_state': 0,
            'train_acc_crop': 0, 'train_acc_state': 0,
            'val_loss_crop': 0, 'val_loss_state': 0,
            'val_acc_crop': 0, 'val_acc_state': 0
        }
        
        fold_pbar = tqdm(enumerate(kf.split(self.dataset.train_samples)), 
                         total=self.config['training']['cv_folds'], 
                         desc="Folds", leave=False)
        
        for fold_idx, (train_idx, val_idx) in fold_pbar:
            
            train_loader, val_loader = self._get_data_loaders(train_idx, val_idx)
            
            fold_train_metrics = self.train_fold(fold_idx, train_loader)
            fold_val_metrics = self.validate_fold(fold_idx, val_loader)
            
            for key in epoch_metrics:
                if 'train' in key:
                    epoch_metrics[key] += fold_train_metrics[key]
                elif 'val' :
                    epoch_metrics[key] += fold_val_metrics[key]
                else:
                    raise ValueError(f"Unknown key: {key}")
            
            fold_pbar.set_postfix({
                'Train Loss Crop': f"{fold_train_metrics['train_loss_crop'] * 100:.2f}%",
                'Train Acc Crop': f"{fold_train_metrics['train_acc_crop'] * 100:.2f}%",
                'Val Loss Crop': f"{fold_train_metrics['val_loss_crop'] * 100:.2f}%",
                'Val Acc Crop': f"{fold_train_metrics['val_acc_crop'] * 100:.2f}%"
            })
        
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
                          desc="Batches", leave=False)
        
        for batch_idx, batch in batch_pbar:
            crop_labels = batch['crop_label']
            crop_label_idx = batch['crop_idx']
            idxs = batch['idx']
            img_paths = batch['img_path']
            splits = batch['split']
            state_labels = batch['state_label']
            state_label_idx = batch['state_idx']
            
            # Load batch of images
            images = []
            for path, split in zip(img_paths, splits):
                images.append(self.dataset.load_image_from_path(path, split))
            
            images_tensor = torch.stack(images, dim=0)
            
            
            batch_metrics = self.train_batch(batch_idx, images_tensor, crop_label_idx, state_label_idx)
            
            train_loss_crop += batch_metrics['loss_crop']
            train_loss_state += batch_metrics['loss_state']
            train_correct_crop += batch_metrics['correct_crop']
            train_correct_state += batch_metrics['correct_state']
            train_total += batch_metrics['total']
            
            batch_pbar.set_postfix({
                'TLC': f"{batch_metrics['loss_crop'] * 100:.2f}%",
                'TLS': f"{batch_metrics['loss_state'] * 100:.2f}%",
                'TAC': f"{(batch_metrics['correct_crop'] / batch_metrics['total']) * 100:.2f}%",
                'TAS': f"{(batch_metrics['correct_state'] / batch_metrics['total']) * 100:.2f}%",
                'Total': f"{batch_metrics['total']}"
            })
        
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
        loss = loss_crop + loss_state
        
        # Backward pass and optimise
        loss.backward()
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
                          desc="Batches", leave=False)
        
        for batch_idx, batch in batch_pbar:
            crop_labels = batch['crop_label']
            crop_label_idx = batch['crop_idx']
            idxs = batch['idx']
            img_paths = batch['img_path']
            splits = batch['split']
            state_labels = batch['state_label']
            state_label_idx = batch['state_idx']
            
            # Load batch of images
            images = []
            for path, split in zip(img_paths, splits):
                images.append(self.dataset.load_image_from_path(path, split))
            
            images_tensor = torch.stack(images, dim=0)
            
            batch_metrics = self.validate_batch(batch_idx, images_tensor, crop_label_idx, state_label_idx)
            
            val_loss_crop += batch_metrics['loss_crop']
            val_loss_state += batch_metrics['loss_state']
            val_correct_crop += batch_metrics['correct_crop']
            val_correct_state += batch_metrics['correct_state']
            val_total += batch_metrics['total']
            
            batch_pbar.set_postfix({
                'VLC': f"{batch_metrics['loss_crop'] * 100:.2f}%",
                'VLS': f"{batch_metrics['loss_state'] * 100:.2f}%",
                'VAC': f"{(batch_metrics['correct_crop'] / batch_metrics['total']) * 100:.2f}%",
                'VAS': f"{(batch_metrics['correct_state'] / batch_metrics['total']) * 100:.2f}%",
                'Total': f"{batch_metrics['total']}"
            })
            
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
        test_loader = DataLoader(self.dataset.test_samples, batch_size=self.config['evaluation']['batch_size'])
        test_loss, test_acc_crop, test_acc_state = self.validate(test_loader)
        print(f"Test Loss: {test_loss:.4f}, Crop Acc: {test_acc_crop:.4f}, State Acc: {test_acc_state:.4f}")

    def save_model(self):
        os.makedirs(self.config['output_dir'], exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.config['output_dir'], f"{self.config['model']['name']}.pth"))
        print(f"Model saved to {self.config['output_dir']}")
        