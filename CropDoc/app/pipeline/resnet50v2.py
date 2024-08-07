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
        dataset_manager = DatasetManager(dataset_path, transform=transformer_manager.transformers)
    except Exception as e:
        logger.error(f"Error initialising dataset manager: {e}")
        raise e
    
    # Initialise pipeline
    try:
        pipeline = ResNet50v2Pipeline(model_config, dataset_manager)
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
    
    # Test the model
    try:
        pipeline.test()
    except Exception as e:
        logger.error(f"Error testing pipeline: {e}")
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
    def __init__(self, config, dataset: DatasetManager):
        logger.debug("Initialising ResNet50v2 Pipeline")
        
        # Config used to configure the pipeline
        self.config = config
        self.dataset = dataset
        
        # Pipeline main objects
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._create_model()
        
        # Loss functions
        self.criterion_crop, self.criterion_state = self._get_loss_functions()
        
        # Optimisers
        self.batch_optimizer = self._get_optimizer()
        self.epoch_optimizer = self._get_optimizer()
        
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

    def _get_data_loaders(self, train_indices, val_indices):
        train_dataset = torch.utils.data.Subset(self.dataset.train_samples, train_indices)
        val_dataset = torch.utils.data.Subset(self.dataset.train_samples, val_indices)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['workers'],
            pin_memory=True if torch.cuda.is_available() else False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['workers'],
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        return train_loader, val_loader

    def _get_terminal_width(self):
        return int(shutil.get_terminal_size().columns * 0.9)
    
    def train(self):
        logger.debug("Training ResNet50v2 Model")
        k_folds = self.config['training']['cv_folds']
        if k_folds > 1:
            kf = KFold(n_splits=self.config['training']['cv_folds'], shuffle=True, random_state=42)
        else:
            full_range = range(len(self.dataset.train_samples))
            kf = [(full_range, full_range)]
        
        epoch_pbar = trange(int(self.config['training']['epochs']), desc="Epochs", ncols=max(120, self.terminal_width))
        for epoch in epoch_pbar:
            epoch_metrics = self.train_epoch(kf)
            
            # Add metrics to plot later
            self.epoch_train_crop_loss.append(epoch_metrics['train_loss_crop'])
            self.epoch_train_state_loss.append(epoch_metrics['train_loss_state'])
            self.epoch_val_crop_loss.append(epoch_metrics['val_loss_crop'])
            self.epoch_val_state_loss.append(epoch_metrics['val_loss_state'])
            
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
        
        self.epoch_optimizer.zero_grad()
        
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
                total_train_val = fold_train_metrics['train_total'] + fold_val_metrics['val_total']
                train_val_ratio = f"{(fold_train_metrics['train_total'] / total_train_val) * 100:.0f} / {(fold_val_metrics['val_total'] / total_train_val) * 100:.0f}"
                fold_pbar.set_postfix({
                    'Crop Loss Train(Val)': f"{fold_train_metrics['train_loss_crop']:.4f}({fold_val_metrics['val_loss_crop']:.4f})",
                    'State Loss Train(Val)': f"{fold_train_metrics['train_loss_state']:.4f}({fold_val_metrics['val_loss_state']:.4f})",
                    'Crop Acc Train(Val)': f"{fold_train_metrics['train_acc_crop'] * 100:.2f}%({fold_val_metrics['val_acc_crop'] * 100:.2f}%)",
                    'State Acc Train(Val)': f"{fold_train_metrics['train_acc_state'] * 100:.2f}%({fold_val_metrics['val_acc_state'] * 100:.2f}%)",
                    'Total Train(Val)': f"{fold_train_metrics['train_total']}({fold_val_metrics['val_total']}) = {train_val_ratio}"
                })
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
        
        # Calculate loss for crop and state
        loss_crop = epoch_metrics['train_loss_crop'] / num_folds
        loss_state = epoch_metrics['train_loss_state'] / num_folds
        
        # Compute the total loss using criterion
        loss_crop = self.criterion_crop(loss_crop)
        loss_state = self.criterion_state(loss_state)
        
        # Backward pass and optimise
        loss_crop.backward()
        loss_state.backward()
        self.epoch_optimizer.step()
        
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
            
            batch_pbar.set_postfix({
                'Loss Crop': f"{batch_metrics['train_loss_crop']:.4f}",
                'Loss State': f"{batch_metrics['train_loss_state']:.4f}",
                'Acc Crop': f"{(batch_metrics['train_correct_crop'] / batch_metrics['train_total']) * 100:.2f}%",
                'Acc State': f"{(batch_metrics['train_correct_state'] / batch_metrics['train_total']) * 100:.2f}%"
            })

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
        # self.optimizer.zero_grad()
        self.batch_optimizer.zero_grad()
        
        # Forward pass
        model_outputs = self.model(inputs)
        
        crop_outputs = model_outputs[:, :len(self.dataset.unique_crops)]
        state_outputs = model_outputs[:, len(self.dataset.unique_states):]

        # Calculate loss
        loss_crop = self.criterion_crop(crop_outputs, crop_labels)
        loss_state = self.criterion_state(state_outputs, state_labels)
        # loss = loss_crop + loss_state
        
        # Backward pass and optimise
        # loss.backward()
        loss_crop.backward(retain_graph=True)
        loss_state.backward(retain_graph=True)
        self.batch_optimizer.step()
        # self.optimizer.step()
        
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
            
            batch_pbar.set_postfix({
                'Loss Crop': f"{batch_metrics['val_loss_crop']:.4f}",
                'Loss State': f"{batch_metrics['val_loss_state']:.4f}",
                'Acc Crop': f"{(batch_metrics['val_correct_crop'] / batch_metrics['val_total']) * 100:.2f}%",
                'Acc State': f"{(batch_metrics['val_correct_state'] / batch_metrics['val_total']) * 100:.2f}%"
            })
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
        self.model.eval()
        test_loss_crop = 0
        test_loss_state = 0
        test_correct_crop = 0
        test_correct_state = 0
        test_total = 0

        test_loader = torch.utils.data.DataLoader(
            self.dataset.test_samples,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['workers'],
            pin_memory=True if torch.cuda.is_available() else False
        )

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
        output_dir = self.config['pipeline']['output_dir']
        model_dir = os.path.join(output_dir, 'trained_model')
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'resnet50v2_trained.pth')
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Trained model saved to {model_path}")

    def save_tested_model(self):
        output_dir = self.config['pipeline']['output_dir']
        model_dir = os.path.join(output_dir, 'tested_model')
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'resnet50v2_tested.pth')
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Tested model saved to {model_path}")

    def plot_training_validation_error(self):
        output_dir = self.config['pipeline']['output_dir']
        plot_dir = os.path.join(output_dir, 'plots')
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
        
