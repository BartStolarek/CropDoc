import os
import matplotlib.pyplot as plt
from loguru import logger
import numpy as np

class EvaluationManager():
    
    def __init__(self, config, output_directory, model_manager):
        
        self.eval_output_dir = os.path.join(output_directory, 'evaluation')
        
        # if it doesn't exist, create the directory
        if not os.path.exists(self.eval_output_dir):
            os.makedirs(self.eval_output_dir, exist_ok=True)
        
        self.model_manager = model_manager
        self.output_directory = output_directory
        self.config = config
        
        # Get the progression metrics
        self.progression_metrics = self.model_manager.model_meta.progression_metrics
        self.performance_metrics = self.model_manager.model_meta.performance_metrics
        
        # Obtained number of epochs
        try:
            self.trained_epochs = [performance_metric.epoch for performance_metric in self.progression_metrics]
        except TypeError as e:
            logger.error(f"TypeError: {e}. Progression metrics is: type ({type(self.progression_metrics)}){self.progression_metrics}")
            raise TypeError(e)
        
        
        # Get a list of losses and accuracies for training and validation
        try:
            self.train_crop_losses = [performance_metric.train.crop.loss for performance_metric in self.progression_metrics]
        except TypeError as e:
            logger.error(f"TypeError: {e}. Progression metrics is: type ({type(self.progression_metrics)}){self.progression_metrics}")
            raise TypeError(e)
        self.train_state_losses = [performance_metric.train.state.loss for performance_metric in self.progression_metrics]
        self.train_crop_accuracies = [performance_metric.train.crop.accuracy for performance_metric in self.progression_metrics]
        self.train_state_accuracies = [performance_metric.train.state.accuracy for performance_metric in self.progression_metrics]
        
        self.val_crop_losses = [performance_metric.val.crop.loss for performance_metric in self.progression_metrics]
        self.val_state_losses = [performance_metric.val.state.loss for performance_metric in self.progression_metrics]
        self.val_crop_accuracies = [performance_metric.val.crop.accuracy for performance_metric in self.progression_metrics]
        self.val_state_accuracies = [performance_metric.val.state.accuracy for performance_metric in self.progression_metrics]
        
        # Create evaluation line graphs
        tcl_vs_vcl_graph = self._generate_line_graph(
            self.trained_epochs, {
                'Train Crop Loss': self.train_crop_losses,
                'Validation Crop Loss': self.val_crop_losses,
                'Train State Loss': self.train_state_losses,
                'Validation State Loss': self.val_state_losses
            },
            'Epochs',
            'Loss',
            'Train Loss vs Validation Loss'
        )
        tcl_vs_vcl_graph.savefig(os.path.join(self.eval_output_dir, 'train_loss_vs_val_loss.png'))
        
        tca_vs_vca_graph = self._generate_line_graph(
            self.trained_epochs, {
                'Train Crop Accuracy': self.train_crop_accuracies,
                'Validation Crop Accuracy': self.val_crop_accuracies,
                'Train State Accuracy': self.train_state_accuracies,
                'Validation State Accuracy': self.val_state_accuracies
            },
            'Epochs',
            'Accuracy',
            'Train Accuracy vs Validation Accuracy'
        )
        tca_vs_vca_graph.savefig(os.path.join(self.eval_output_dir, 'train_acc_vs_val_acc.png'))
        
        # Create evaluation confusion matrix for test
        crop_confusion_matrix = self._generate_confusion_matrix(
            confusion_matrix=self.performance_metrics.test.crop.confusion_matrix,
            title='Test Crop Confusion Matrix',
            class_names=self.model_manager.model_meta.crops
        )
        crop_confusion_matrix.savefig(os.path.join(self.eval_output_dir, 'test_crop_confusion_matrix.png'))
        
        state_confusion_matrix = self._generate_confusion_matrix(
            confusion_matrix=self.performance_metrics.test.state.confusion_matrix,
            title='Test State Confusion Matrix',
            class_names=self.model_manager.model_meta.states
        )
        state_confusion_matrix.savefig(os.path.join(self.eval_output_dir, 'test_state_confusion_matrix.png'))
        
    def _generate_confusion_matrix(self, confusion_matrix: list, title: str, class_names: list):
        """
        Generates a confusion matrix plot with class names on axes.

        Args:
            confusion_matrix (list): The confusion matrix to plot.
            title (str): The title of the graph.
            class_names (list): List of class names to use for axis labels.

        Returns:
            plt: The matplotlib.pyplot object with the plot.
        """
        
        confusion_matrix = np.array(confusion_matrix)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
        ax.figure.colorbar(im, ax=ax)
        
        # Set up axes
        ax.set(xticks=np.arange(confusion_matrix.shape[1]),
            yticks=np.arange(confusion_matrix.shape[0]),
            xticklabels=class_names, yticklabels=class_names,
            title=title,
            ylabel='True label',
            xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        thresh = confusion_matrix.max() / 2.
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                ax.text(j, i, format(confusion_matrix[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if confusion_matrix[i, j] > thresh else "black")

        fig.tight_layout()
        return plt
        
    
    
    def _generate_line_graph(self, x_values: list, y_values: dict, x_label: str, y_label: str, title: str):
        """
        Generates a line graph comparing multiple y-values (e.g., train and validation loss) over the given x-values.

        Args:
            x_values (list): The x-axis values (e.g., epochs).
            y_values (dict): A dictionary where keys are labels (e.g., 'Train Loss', 'Validation Loss') and values are lists of y-axis values.
            x_label (str): The label for the x-axis.
            y_label (str): The label for the y-axis.
            title (str): The title of the graph.

        Returns:
            plt: The matplotlib.pyplot object with the plot.
        """
        
        plt.figure(figsize=(10, 6))
        
        for label, y in y_values.items():
            plt.plot(x_values, y, label=label)
        
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        return plt
