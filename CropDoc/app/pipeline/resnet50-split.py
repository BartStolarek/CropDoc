import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import os
from loguru import logger
from typing import List
import re
from PIL import Image
from pprint import pprint
import warnings
from sklearn.model_selection import KFold
import numpy as np
import time
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.parallel.parallel_apply")


class TransformerManager():
    def __init__(self, transformers: dict[str, list[dict]]):
        self.transformers = self._get_transformers(transformers)
        
    def __len__(self):
        return self._calculate_total_transforms()
    
    def _calculate_total_transforms(self):
        total_transforms = 0
        for split, transformer in self.transformers.items():
            total_transforms += len(transformer)
        return total_transforms
    
    def _get_transformers(self, transformers: dict[str, list[dict]]) -> dict[str, torchvision.transforms.Compose]:
        logger.debug("Getting transformers")
        if not transformers:
            logger.warning("No transformers provided")
            return None
        
        if not isinstance(transformers, dict):
            logger.error("Transformers must be a dictionary: provided a {type(transforms)} : {transforms}")
            return None
        
        if not set(['train', 'val', 'test']).issubset(transformers.keys()):
            logger.error("Transformers must contain keys 'train', 'val', and 'test'")
            return None
        
        # Get train transformers
        train_transformers = self._build_transformers(transformers['train'])
        
        # Get val transformers
        val_transformers = self._build_transformers(transformers['val'])
        
        # Get test transformers
        test_transformers = self._build_transformers(transformers['test'])
        
        transformer_dict = {'train': train_transformers, 'val': val_transformers, 'test': test_transformers}
        
        logger.info("Created transformers dict")
        
        return transformer_dict
    
    def _build_transformers(self, transformer_list: List[dict]) -> torchvision.transforms.Compose:
        logger.debug("Building transformers")
        if not transformer_list:
            logger.warning("No transformers provided")
            return None
        
        if not isinstance(transformer_list, list):
            logger.error("Transformers must be a list")
            return None
        
        transforms_list = []
        for transform_dict in transformer_list:
            try:
                transform = self._parse_transform(transform_dict)
                if transform:
                    transforms_list.append(transform)
            except Exception as e:
                logger.error(f"Error parsing transform '{transform_dict}': {e}")
        
        logger.info(f"Built transformers list {transforms_list}")
        return torchvision.transforms.Compose(transforms_list)
    
    def _parse_transform(self, transform_dict: dict):
        logger.debug(f"Parsing transform {transform_dict}")
        transform_name = transform_dict['type']
        
        # Remove type and its value from dict
        transform_dict.pop('type')
        
        transform_fn = getattr(torchvision.transforms, transform_name)
        logger.info(f"Converted transformer dict to {transform_fn}")
        return transform_fn(**transform_dict)
        
    
class CropCCMTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, transformers=None, split='train'):
        if not isinstance(transformers, torchvision.transforms.Compose):
            logger.error(f"Invalid transforms: {transformers}. Must be of type {torchvision.transforms.Compose}")
            raise ValueError(f"Invalid transforms: {transformers}. Must be of type {torchvision.transforms.Compose}")
        
        self.root = dataset_path
        self.transforms = transformers
        self.split = split
        
        self.data_map = {
            'crop': {},
            'state': {}
        }
        
        self.images = np.empty(0, dtype=object)
        self.labels = np.empty((0, 2), dtype=int)  # Two columns for crop and state labels
        
        splits = ['train', 'test']
        if split not in splits:
            raise ValueError(f"Invalid split: {split}. Must be one of {splits}")

        self.walk_through_root_append_images_and_labels()
               
        self.crops = self._get_unique_crops()
        self.states = self._get_unique_states()
        
    
    def get_unique_crop_count(self):
        return len(self.crops)
    
    def get_unique_state_count(self):
        return len(self.states)
    
    def equal_reduce(self, reduce_to: float):
        
        images = []
        labels = []
        for state, count in self.data_map['state'].items():
            reduction_amount = int(count * reduce_to)
            added = 0
            for i, img_path in enumerate(self.images):
                state_check = state
                if 'healthy' in state_check:
                    crop = state_check.split('-')[0]
                    if crop in img_path and 'healthy' in img_path:
                        images.append(self.images[i])
                        labels.append(self.labels[i])
                        added += 1
                elif state_check in img_path:
                    images.append(self.images[i])
                    labels.append(self.labels[i])
                    added += 1
                    
                if added >= reduction_amount:
                    break
        self.images = images
        self.labels = labels
        self.crops = self._get_unique_crops()
        self.states = self._get_unique_states()
        
        logger.info(f'Reduced dataset to {len(self.images)} images')
                
    
    def walk_through_root_append_images_and_labels(self):
        logger.debug(f"Walking through root path: {self.root} to obtain images and labels")
        # Walk through root path
        
        images = []
        labels = []
        
        for root, directory, files in os.walk(self.root):
            for file in files:
                file_name = os.path.basename(file)
                
                # Remove extension
                file_name = os.path.splitext(file_name)[0]
                
                # Remove file id from start of file name
                # file_id = re.match(r'^\d+', file_name)
                file_name = re.sub(r'^\d+', '', file_name)
                
                # Split file name into classes and split
                classes_and_split = file_name.split('_')
                
                # Obtain crop label and split directory
                crop_label = classes_and_split[0]
                split_directory = classes_and_split[1]
                
                # Change split directory to be consistent
                if split_directory == 'valid':
                    split_directory = 'test'
                
                # Check if required split matches
                if split_directory != self.split:
                    continue
                
                # Obtain state label
                state_label = classes_and_split[2]
                
                # Change healthy state labels to include crop label in them to differentiate
                if state_label == 'healthy':
                    state_label = crop_label + '-healthy'
                
                # Append image path and labels
                images.append(os.path.join(root, (os.path.basename(file))))
                labels.append((crop_label, state_label))
                
                if crop_label not in self.data_map['crop']:
                    self.data_map['crop'][crop_label] = 1
                else:
                    self.data_map['crop'][crop_label] += 1

                if state_label not in self.data_map['state']:
                    self.data_map['state'][state_label] = 1
                else:
                    self.data_map['state'][state_label] += 1
                    
        self.images = np.array(images, dtype=object)
        self.labels = np.array(labels, dtype=object)
                    
        logger.info(f"Crop Data Map: {self.data_map['crop']}")
        logger.info(f"State Data Map: {self.data_map['state']}")
                 
    def _get_unique_crops(self):
        crops = []
        for crop, _ in self.labels:
            if crop not in crops:
                crops.append(crop)
        crops = sorted(list(set(crops)))
        return crops
    
    def _get_unique_states(self):
        states = []
        for _, state in self.labels:
            if state not in states:
                states.append(state)
        states = sorted(list(set(states)))
        return states

    def __getitem__(self, idx):
        img_path = self.images[idx]
        crop, state = self.labels[idx]

        img = Image.open(img_path).convert('RGB')

        # Convert crop and state to numeric labels
        crop_label = self.crops.index(crop)
        state_label = self.states.index(state)

        if self.transforms:
            img = self.transforms(img)
        
        return img, crop_label, state_label

    def __len__(self):
        return len(self.images)


class MultiHeadResNetModel(torch.nn.Module):
    def __init__(self, num_classes_crop, num_classes_state):
        super(MultiHeadResNetModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        num_ftres = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Identity()
        self.crop_fc = torch.nn.Linear(num_ftres, num_classes_crop)
        self.state_fc = torch.nn.Linear(num_ftres, num_classes_state)
        
        # Move all parts of the model to the same device
        self.resnet = self.resnet.to(self.device)
        self.crop_fc = self.crop_fc.to(self.device)
        self.state_fc = self.state_fc.to(self.device)
        
        # Wrap only the resnet part in DataParallel
        self.resnet = torch.nn.DataParallel(self.resnet)
    
    def forward(self, x):
        x = x.to(self.device)  # Move input to GPU if available
        x = self.resnet(x) #  x.shape = torch.Size(<batch_size>, <num_features>)
        crop_out = self.crop_fc(x)
        state_out = self.state_fc(x)
        return crop_out, state_out


class Pipeline():
    def __init__(self, config, dataset=None):
        # Parse config
        try:
            self.model_config = config['model']
            self.pipeline_config = config['pipeline']
        except Exception as e:
            logger.error(f"Error parsing config: {e}")
            raise ValueError(f"Error parsing config: {e}")
        
        self.dataset_root = self.pipeline_config['dataset_dir'] if dataset is None else dataset
        
        self.output_dir = self.pipeline_config['output_dir']
        
        # Load transformers
        self.transformer_manager = TransformerManager(self.model_config['data']['transformers'])

        # Load the data
        self.train_data = CropCCMTDataset(dataset_path=self.dataset_root, transformers=self.transformer_manager.transformers['train'], split='train')
        self.test_data = CropCCMTDataset(dataset_path=self.dataset_root, transformers=self.transformer_manager.transformers['test'], split='test')
        
        dev_reduce = self.model_config['data']['reduce']
        
        if dev_reduce < 1.0:
            self.train_data.equal_reduce(dev_reduce)
            self.test_data.equal_reduce(dev_reduce)
        
        # Define the model 
        self.model = MultiHeadResNetModel(num_classes_crop=self.train_data.get_unique_crop_count(), num_classes_state=self.train_data.get_unique_state_count())
        
        # Define the loss function
        self.crop_criterion = torch.nn.CrossEntropyLoss()
        self.state_criterion = torch.nn.CrossEntropyLoss()
        
        # Define the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
        # Define learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.model_config['training']['lr_scheduler']['step_size'],
            gamma=self.model_config['training']['lr_scheduler']['gamma']
        )

    def _get_dataloader(self, dataset, shuffle=True):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.model_config['training']['batch_size'],
            shuffle=shuffle,
            num_workers=self.model_config['training']['num_workers'],
            pin_memory=True
        )
        return dataloader
    
    def train(self):
        logger.debug('Initiated training...')
        epochs = self.model_config['training']['epochs']
        train_metrics = []
        for epoch in range(epochs):
            epoch_metrics = self.train_one_epoch(idx=epoch)
            
            # Update the learning rate
            self.scheduler.step()
            
            logger.info(f'Epoch {epoch} Metrics: {self.format_metrics(epoch_metrics)}')
            train_metrics.append(epoch_metrics)
        
        logger.info('Finished training')
                
    def train_one_epoch(self, idx):
        logger.debug(f"Training epoch {idx}")
        epoch_metrics = self.split_train_validate()
        
        torch.cuda.empty_cache()
        return epoch_metrics
        
    def split_train_validate(self):
        # Create a random subset for training and validation
        train_dataset, val_dataset = torch.utils.data.dataset.random_split(
            self.train_data,
            [1 - self.model_config['training']['val_split'], self.model_config['training']['val_split']]
        )
        
        logger.info(f'Train dataset size: {len(train_dataset)}')
        logger.info(f'Validation dataset size: {len(val_dataset)}')
        
        # Create data loaders with random split
        train_loader = self._get_dataloader(dataset=train_dataset, shuffle=True)
        val_loader = self._get_dataloader(dataset=val_dataset, shuffle=False)
        
        # Set model to training mode
        self.model.train()
        
        # Train the model using train loader
        train_metrics = self.feed_model(train_loader, train=True)
        
        logger.info(f'Train metrics: {self.format_metrics(train_metrics)}')
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Turn off gradients for validation
        with torch.no_grad():
            # Validate the model using val loader
            val_metrics = self.feed_model(val_loader)
        
        logger.info(f'Validation metrics: {self.format_metrics(val_metrics)}')
    
        return {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
              
    
    def feed_model(self, data_loader, train=False):
        loss_crop_total = 0
        loss_state_total = 0
        correct_crop = 0
        correct_state = 0
        total = 0
        
        average_time = []
        
        for i, (images, crop_labels, state_labels) in enumerate(data_loader):
            
            start_time = time.time()
            
            #print(f'Batch {i}/{len(data_loader)}')
            if train:
                self.optimizer.zero_grad()
                
            # Move data to GPU if available
            images = images.to(self.model.device)
            crop_labels = crop_labels.to(self.model.device)
            state_labels = state_labels.to(self.model.device)

            # Forward pass
            crop_predictions, state_predictions = self.model(images)  # (torch.Size(<batch_size>, <num_classes>), torch.Size(<batch_size>, <num_classes>))
            
            # Calculate loss
            loss_crop = self.crop_criterion(crop_predictions, crop_labels)  # torch.Tensor(<float>, gradient_function=<>)
            loss_state = self.state_criterion(state_predictions, state_labels)  # torch.Tensor(<float>, gradient_function=<>)
            loss = loss_crop + loss_state  # torch.Tensor(<float>, gradient_function=<>) + torch.Tensor(<float>, gradient_function=<>)
            
            if train:
                # Backward pass
                loss.backward()
                self.optimizer.step()
            
            loss_crop_total += loss_crop.item()  # Obtain just the float value and add it to the total loss
            loss_state_total += loss_state.item()
            
            # Calculate correct predictions
            _, predicted_crop = torch.max(crop_predictions, 1)
            _, predicted_state = torch.max(state_predictions, 1)
            correct_crop += (predicted_crop == crop_labels).sum().item()
            correct_state += (predicted_state == state_labels).sum().item()
            total += crop_labels.size(0)
            
            end_time = time.time()
            duration = end_time - start_time
            average_time.append(duration)
            
        print(f'Avg Time: {np.mean(average_time)}, estimated total time in minutes: {np.mean(average_time) * len(data_loader) / 60}')
            
        return {
            'loss_crop': loss_crop_total / len(data_loader),
            'loss_state': loss_state_total / len(data_loader),
            'correct_crop': correct_crop,
            'correct_state': correct_state,
            'accuracy_crop': 100 * correct_crop / total,
            'accuracy_state': 100 * correct_state / total,
            'total': total
        }
           
    def save_model(self):
        model_output_dir = 'model'
        full_model_output_dir = os.path.join(self.output_dir, model_output_dir)
        os.makedirs(full_model_output_dir, exist_ok=True)
        model_name = f"{self.model_config['name']}-{self.model_config['version']}.pth"
        model_output_path = os.path.join(full_model_output_dir, model_name)
        torch.save(self.model.state_dict(), model_output_path)
    
    def load_model(self):
        model_name = f"{self.model_config['name']}-{self.model_config['version']}.pth"
        model_path = os.path.join(self.output_dir, 'model', model_name)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.to(self.model.device)
        self.model.eval()
        
    def format_metrics(self, metrics):
        formatted_metrics = {}
        for key, value in metrics.items():
            try:
                if isinstance(value, float):
                    if 'accuracy' in key:
                        formatted_metrics[key] = f"{value:.2f}%"
                    else:
                        formatted_metrics[key] = f"{value:.2f}"
                elif isinstance(value, dict):
                    formatted_metrics[key] = self.format_metrics(value)
            except Exception as e:
                logger.error(f"Error formatting metric {key}: {e}")
                raise e
        return formatted_metrics
    
    def predict_test_dataset(self):
        test_loader = self._get_dataloader(self.test_data, shuffle=False)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Turn off gradients for validation
        with torch.no_grad():
            # Validate the model using val loader
            test_metrics = self.feed_model(test_loader)
        
        logger.info(f'Test metrics: {self.format_metrics(test_metrics)}')
    
    def predict_single_image_from_dataset(self, idx):
        image, _, _ = self.train_data[idx]
        image = image.unsqueeze(0).to(self.model.device)
        with torch.no_grad():
            crop_prediction, state_prediction = self.model(image)
        
        _, predicted_crop = torch.max(crop_prediction, 1)
        _, predicted_state = torch.max(state_prediction, 1)
        
        print(f'Predicted Crop: {self.train_data.crops[predicted_crop.item()]}')
        print(f'Predicted State: {self.train_data.states[predicted_state.item()]}')
        print(f'Actual Crop: {self.train_data.crops[0]}')
        print(f'Actual State: {self.train_data.states[0]}')
        
        plt.imshow(image.squeeze().permute(1, 2, 0).cpu().numpy())
        plt.title(f'Predicted Crop: {self.train_data.crops[predicted_crop.item()]}\n Predicted State: {self.train_data.states[predicted_state.item()]}')
        plt.axis('off')
        plt.show()
      
    def predict(self, image_path):
        # Predict a single image
        image = Image.open(image_path).convert('RGB')
        image = self.transformer_manager.transformers['test'](image)
        image = image.unsqueeze(0).to(self.loaded_model.device)
        with torch.no_grad():
            crop_prediction, state_prediction = self.loaded_model(image)
            
        # Get the predicted classes
        _, predicted_crop = torch.max(crop_prediction, 1)
        _, predicted_state = torch.max(state_prediction, 1)
        
        print(f'Predicted Crop: {self.train_data.crops[predicted_crop.item()]}')
        print(f'Predicted State: {self.train_data.states[predicted_state.item()]}')
        
        plt.imshow(image.squeeze().permute(1, 2, 0).cpu().numpy())
        plt.title(f'Predicted Crop: {self.train_data.crops[predicted_crop.item()]}\n Predicted State: {self.train_data.states[predicted_state.item()]}')
        plt.axis('off')
        plt.show()
    
    def save_metrics(self, metrics:dict):
        # Save to a txt file in the output/evaluation directory
        metrics_output_dir = 'evaluate'
        full_metrics_output_dir = os.path.join(self.output_dir, metrics_output_dir)
        
        


def train(config, dataset=None):
    logger.debug(f'Training')
    
    # Initialise Pipeline
    pipeline = Pipeline(dataset=dataset, config=config)
    
    # Train the model
    metrics = pipeline.train()
    pipeline.save_metrics(metrics)
    
    # Save the trained model
    pipeline.save_model()
    
    logger.info('Training complete')


def predict(config, dataset=None, model_path=None):
    logger.debug(f'Predicting with dataset: {dataset} and model: {model_path}')
    pipeline = Pipeline(dataset=dataset, config=config)
    
    # Load the saved model
    pipeline.load_model()
    
    # Predict the test dataset
    pipeline.predict_test_dataset()
    
    logger.info('Prediction complete')
    return pipeline

def predict_one(config, dataset=None, model_path=None):
    logger.debug('Predicting')
    pipeline = Pipeline(dataset=dataset, config=config)
    
    # Load the saved model
    pipeline.load_model()
    
    # Predict a single image from the test dataset
    random_idx = np.random.randint(0, len(pipeline.train_data))
    pipeline.predict_single_image_from_dataset(random_idx)
    
    logger.info('Prediction complete')
    return pipeline