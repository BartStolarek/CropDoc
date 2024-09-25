from app.pipeline_helper.dataloader import TransformDataLoader
from app.pipeline_helper.metricstrackers import PerformanceMetrics, ProgressionMetrics, SplitMetrics, Metrics
from app.pipeline_helper.pipelinemanager import PipelineManager
import torch
from loguru import logger
import shutil
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

class TrainingManager(PipelineManager):
    def __init__(self,
                 config,
                 output_directory,
                 train_data,
                 train_transformers,
                 val_transformers,
                 model_manager,
                 crop_criterion,
                 state_criterion,
                 optimiser,
                 scheduler):
        super().__init__()
        self.output_directory = output_directory
        self.config = config
        
        self.epochs_to_train = int(config['train']['epochs'])
        self.checkpoint_interval = int(config['train']['checkpoint_interval'])
        self.validation_split = float(config['train']['validation_split'])
        self.batch_size = int(config['train']['batch_size'])
        self.num_workers = int(config['train']['num_workers'])
        self.model_manager = model_manager
        self.model = model_manager.get_model()
        self.crop_criterion = crop_criterion
        self.state_criterion = state_criterion
        self.optimiser = optimiser
        self.scheduler = scheduler
        
        
        self.train_dataset, self.val_dataset = torch.utils.data.dataset.random_split(
            dataset=train_data,
            lengths=[
                1 - self.validation_split,
                self.validation_split
            ]
        )
        
        self.train_dataloader = TransformDataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            transform=train_transformers
        )
        
        self.val_dataloader = TransformDataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            transform=val_transformers
        )
        logger.info(f"Training Manager initialised with {len(self.train_dataset)} training samples and {len(self.val_dataset)} validation samples")
        
    def start_training(self):
        
        # Get the starting epoch
        self.pretrained_epochs = self.model_manager.get_number_of_trained_epochs()
        self.starting_epoch = self.pretrained_epochs + 1
        self.ending_epoch = self.starting_epoch + self.epochs_to_train
        logger.info(f"Starting training loop for epochs {self.starting_epoch} to {self.ending_epoch - 1} ({self.ending_epoch - self.starting_epoch}). Total pre-trained epochs {self.pretrained_epochs}")
        
        logger.info('\nTraining loop index:\n' + "T: Train, V: Validation\n" +
                    "C: Crop, S: State\n" + "L: Loss, A: Accuracy")
        
        # Initialise epoch progress bar and training metrics list
        self.terminal_width = shutil.get_terminal_size().columns
        epochs_progress = tqdm(range(self.starting_epoch, self.ending_epoch),
                               desc="Epoch",
                               leave=True,
                               ncols=int(self.terminal_width * 0.99))
        
        # Obtain progression metric tracker
        self.progression_metrics = self.model_manager.get_progress_metrics_tracker()
        if not isinstance(self.progression_metrics, ProgressionMetrics):
            raise TypeError(f"Progression metrics is not of type ProgressionMetrics, instead got {type(self.progression_metrics)}")
        
        self.performance_metrics = self.model_manager.get_performance_metrics_tracker()
        
        
        # Set the checkpoint start
        if self.pretrained_epochs > 0:
            last_checkpointed_epoch = self.starting_epoch - 1
            initial_epoch = False
        else:
            initial_epoch = True
        
        
         
        # Start the training loop
        for i in epochs_progress:  # TODO: Add to the report the number of epochs we trained for

            # Train the model for one epoch
            epoch_performance_metrics = self._train_one_epoch(epoch=i)
            epoch_performance_metrics.assign_epoch(i)
            
            # Update the learning rate
            val_loss = epoch_performance_metrics.get_combined_loss('val')
            self.scheduler.step(val_loss)

            # Check if its first epoch, if so save a checkpoint
            if initial_epoch:
                last_checkpointed_epoch = i
                self.performance_metrics = epoch_performance_metrics
                self.performance_metrics.assign_epoch(i)
                self._save_checkpoint(epoch=i)

            # Check if the current epoch is far enough from previous checkpoint and also past quarter way for all epochs
            elif (i - last_checkpointed_epoch
                  ) >= self.checkpoint_interval and i >= self.ending_epoch / 4:
                # Check if current epoch has better validation loss than current best
                if self.check_if_val_metrics_improved(epoch_performance_metrics):
                    self.performance_metrics = epoch_performance_metrics
                    self.performance_metrics.assign_epoch(i)
                    self._save_checkpoint(epoch=i)
                    last_checkpointed_epoch = i
            
            # Check if this is the last epoch
            elif i == self.ending_epoch - 1:
                if self.check_if_val_metrics_improved(epoch_performance_metrics):
                    self.performance_metrics = epoch_performance_metrics
                    self.performance_metrics.assign_epoch(i)
                    self._save_checkpoint(epoch=i)
                    last_checkpointed_epoch = i
            
            # Update the progress bar with the metrics
            epochs_progress.set_postfix(epoch_performance_metrics.get_formatted_metrics_as_dict())

            # Append the metrics to the training metrics list
            self.progression_metrics.append(epoch_performance_metrics)
            if not isinstance(self.progression_metrics, ProgressionMetrics):
                raise TypeError(f"Progression metrics is not of type ProgressionMetrics, instead got {type(self.progression_metrics)}")

            initial_epoch = False

        logger.info(f'Training loop complete, trained model for epochs {self.starting_epoch} to {self.ending_epoch - 1} ({self.ending_epoch - self.starting_epoch})')

        # Update model meta
        self.update_model_meta()

    def _train_one_epoch(self, epoch) -> dict:
        """ Train the model for one epoch, splitting the training data into a training and validation set

        Returns:
            dict: A dictionary containing the performance metrics for this epoch's training and validation
        """

        epoch_metrics = PerformanceMetrics(epoch=epoch)

        
        # Set the model to train mode
        self.model.train()

        # Feed the model the training data, and set feeder to train
        train_metrics = self._feed_model(self.train_dataloader, split='train')
        train_metrics.assign_split('train')
        epoch_metrics.train = train_metrics

        # Set the model to evaluation/validation mode
        self.model.eval()

        # Turn off gradients for validation
        with torch.no_grad():

            # Feed the model the validation data
            val_metrics = self._feed_model(self.val_dataloader, split='val')
            val_metrics.assign_split('val')
            epoch_metrics.val = val_metrics

        # If using GPU clear the cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return epoch_metrics
    
    def _save_checkpoint(self, epoch):
        
        self.model.save_checkpoint(
            epoch=epoch,
            optimiser=self.optimiser,
            scheduler=self.scheduler,
            model_meta=self.model_manager.model_meta,
            checkpoint_directory=self.model_manager.checkpoint_directory,
            filename=self.model_manager.name_version
        )
    
    def check_if_val_metrics_improved(self, new_metrics: PerformanceMetrics):
        """

        Args:
           

        Returns:
            
        """
        # Get the change between best validation metrics and the new metrics for loss
        crop_loss_change = self._get_change_percentage(new_metrics.val.crop.loss, self.performance_metrics.val.crop.loss)
        state_loss_change = self._get_change_percentage(new_metrics.val.state.loss, self.performance_metrics.val.state.loss)
        average_loss_change = abs((crop_loss_change + state_loss_change) / 2)
        
        # Get the change between best validation metrics and the new metrics for accuracy
        crop_accuracy_change = self._get_change_percentage(new_metrics.val.crop.accuracy, self.performance_metrics.val.crop.accuracy)
        state_accuracy_change = self._get_change_percentage(new_metrics.val.state.accuracy, self.performance_metrics.val.state.accuracy)
        average_accuracy_change = abs((crop_accuracy_change + state_accuracy_change) / 2)
        
        # Average out both averages
        average = (average_loss_change + average_accuracy_change) / 2
        
        if average > 0:
            return True
        
        return False
    
    def _get_change_percentage(self, new, previous):
        try:
            return (new / previous) - 1
        except ZeroDivisionError as e:
            logger.debug(f"ZeroDivisionError in _get_change_percentage, new metric: {new} and previous metric: {previous} - {e}")
            return 0
    
    def update_model_meta(self):
        self.model_manager.model_meta.epochs += self.epochs_to_train
        self.model_manager.model_meta.performance_metrics.train = self.performance_metrics.train
        self.model_manager.model_meta.performance_metrics.val = self.performance_metrics.val
        if not isinstance(self.progression_metrics, ProgressionMetrics):
            raise TypeError(f"Progression metrics is not of type ProgressionMetrics, instead got {type(self.progression_metrics)}")
        for metric in self.progression_metrics:
            if not isinstance(metric, PerformanceMetrics):
                raise TypeError(f"Progression metrics is not of type PerformanceMetrics, instead got {type(metric)}")
            
            if not isinstance(self.model_manager.model_meta.progression_metrics, ProgressionMetrics):
                raise TypeError(f"Progression metrics is not of type ProgressionMetrics, instead got {type(self.model_manager.model_meta.progression_metrics)}")
            
            self.model_manager.model_meta.progression_metrics[metric.epoch] = metric
            
        # Remove any additional progression metrics
        self.model_manager.model_meta.progression_metrics = self.model_manager.model_meta.progression_metrics[:self.model_manager.model_meta.epochs]  
                    
