from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from app.pipeline_helper.metricstrackers import PerformanceMetrics, ProgressionMetrics, SplitMetrics, Metrics
from tqdm import tqdm
import shutil
from loguru import logger
import torch

class PipelineManager:
    def _feed_model(self, dataloader, split) -> SplitMetrics:
        """ Feed the model with the data from the dataloader to train/validate/test the model

        Args:
            dataloader (): The DataLoader to feed the model with
            train (bool, optional): Whether the model is training. Defaults to False if validating or testing.

        Returns:
            SplitMetrics: A object containing the performance metrics for this epoch, including a list of batch metrics.
        """
        splits = ['train', 'val', 'test']
        if split not in splits:
            raise ValueError(f"Split must be one of {splits}, but got {split}")
        
        # Initialise metrics for whole epoch (total)
        loss_crop_total = 0
        loss_state_total = 0
        loss_combined_total = 0
        correct_crop_total = 0
        correct_state_total = 0
        count_total = 0

        all_crop_predictions = []
        all_crop_labels = []
        all_state_predictions = []
        all_state_labels = []

        # Initialise metrics for just one dataset split, which will need crop and state metrics
        label_metrics = SplitMetrics(split=split)

        # Create the batch progress bar
        self.terminal_width = shutil.get_terminal_size().columns
        batch_progress = tqdm(enumerate(dataloader),
                              desc="Batch",
                              leave=False,
                              total=len(dataloader),
                              ncols=int(self.terminal_width * 0.99))

        # Start feeding the model in batches
        for i, (images, crop_labels, state_labels) in batch_progress:

            # If training, set the optimiser to zero the gradients because we are about to calculate new gradients
            if split == 'train':
                self.optimiser.zero_grad()

            # Move data to GPU if available
            images = images.to(self.model.device)
            crop_labels = crop_labels.to(self.model.device)
            state_labels = state_labels.to(self.model.device)

            # Forward pass
            crop_predictions, state_predictions = self.model(images)

            # Calculate the loss
            loss_crop_batch = self.crop_criterion(crop_predictions,
                                                  crop_labels)
            loss_state_batch = self.state_criterion(state_predictions,
                                                    state_labels)
            loss_combined_batch = loss_crop_batch + loss_state_batch

            if split == 'train':
                # Backward pass
                loss_crop_batch.backward(retain_graph=True)
                loss_state_batch.backward()
                self.optimiser.step()

            # Calculate correct predictions
            _, predicted_crop = torch.max(crop_predictions, 1)
            _, predicted_state = torch.max(state_predictions, 1)
            correct_crop_batch = (predicted_crop == crop_labels).sum().item()
            correct_state_batch = (
                predicted_state == state_labels).sum().item()
            count_batch = crop_labels.size(0)

            # Store predictions and labels for later confusion matrix
            all_crop_predictions.extend(predicted_crop.cpu().numpy())
            all_crop_labels.extend(crop_labels.cpu().numpy())
            all_state_predictions.extend(predicted_state.cpu().numpy())
            all_state_labels.extend(state_labels.cpu().numpy())

            # Update total epoch metrics
            loss_crop_total += loss_crop_batch.item()
            loss_state_total += loss_state_batch.item()
            loss_combined_total += loss_combined_batch.item()
            correct_crop_total += correct_crop_batch
            correct_state_total += correct_state_batch
            count_total += count_batch

            # TODO: get batch metrics to add them to postfix of batch_progress bar
            # batch_progress.set_postfix(self._format_metrics(batch_metrics))

        # Calculate confusion matrices
        crop_confusion_matrix = confusion_matrix(all_crop_labels,
                                                 all_crop_predictions)
        state_confusion_matrix = confusion_matrix(all_state_labels,
                                                  all_state_predictions)

        # Calculate precision, recall, f1 score
        crop_precision, crop_recall, crop_f1, crop_support = precision_recall_fscore_support(
            all_crop_labels,
            all_crop_predictions,
            average='weighted',
            zero_division=0)
        state_precision, state_recall, state_f1, state_support = precision_recall_fscore_support(
            all_state_labels,
            all_state_predictions,
            average='weighted',
            zero_division=0)
        
        # Create a metrics object for crop
        label_metrics.crop = Metrics(label='crop', split=split)
        label_metrics.crop.loss = loss_crop_total / len(dataloader) if len(dataloader) > 0 else 0
        label_metrics.crop.accuracy = correct_crop_total / count_total if count_total > 0 else 0
        label_metrics.crop.correctly_classified = correct_crop_total
        label_metrics.crop.total_classifications = count_total
        label_metrics.crop.confusion_matrix = crop_confusion_matrix.tolist()
        label_metrics.crop.precision = crop_precision
        label_metrics.crop.recall = crop_recall
        label_metrics.crop.f1_score = crop_f1
        
        # Create a metrics object for state
        label_metrics.state = Metrics(label='state', split=split)
        label_metrics.state.loss = loss_state_total / len(dataloader) if len(dataloader) > 0 else 0
        label_metrics.state.accuracy = correct_state_total / count_total if count_total > 0 else 0
        label_metrics.state.correctly_classified = correct_state_total
        label_metrics.state.total_classifications = count_total
        label_metrics.state.confusion_matrix = state_confusion_matrix.tolist()
        label_metrics.state.precision = state_precision
        label_metrics.state.recall = state_recall
        label_metrics.state.f1_score = state_f1
    
        return label_metrics
