from datetime import datetime
from zoneinfo import ZoneInfo
from dateutil import parser


class Metrics:
    def __init__(self, epoch=None, split=None, label=None, metric_dict=None):
        if metric_dict is None:
            self.epoch = epoch
            self.split = split
            self.label = label
            self.accuracy = None
            self.loss = None
            self.precision = None
            self.recall = None
            self.f1_score = None
            self.correctly_classified = None
            self.total_classifications = None
            self.confusion_matrix = None
        else:
            self.epoch = metric_dict['epoch']
            self.split = metric_dict['split']
            self.label = metric_dict['label']
            self.accuracy = metric_dict['accuracy']
            self.loss = metric_dict['loss']
            self.precision = metric_dict['precision']
            self.recall = metric_dict['recall']
            self.f1_score = metric_dict['f1_score']
            self.correctly_classified = metric_dict['correctly_classified']
            self.total_classifications = metric_dict['total_classifications']
            self.confusion_matrix = metric_dict['confusion_matrix']
        
    def get_formatted_dict(self):
        split_letter = self.split[0].upper()
        label_letter = self.label[0].upper()
        return {
            f"{split_letter}{label_letter}A": self.format_accuracy(self.accuracy),
            f"{split_letter}{label_letter}L": self.format_loss(self.loss),
        }
        
    def format_accuracy(self, metric: float) -> str:
        """ Format the accuracy metric into a percentage string

        Args:
            metric (float): The accuracy metric to format

        Returns:
            str: The formatted accuracy metric as a percentage string
        """
        return f"{metric * 100:.1f}%"

    def format_loss(self, metric: float) -> str:
        """ Format the loss metric to 3 decimal places

        Args:
            metric (float): The loss metric to format

        Returns:
            str: The formatted loss metric to 3 decimal places
        """
        return f"{metric:.3f}"
        
    def change_percentage(self, new_metrics, metric):
        return self._calculate_change(new_metrics[metric], getattr(self, metric))
        
    def _calculate_change(self, new_value: float, old_value: float) -> float:
        """ Calculate the percentage change between two values

        Args:
            new_value (float): The value it is changing to
            old_value (float): The value it is changing from

        Returns:
            float: The percentage change between the two values
        """
        if old_value == 0:
            return 0
        return new_value / old_value - 1
    
    def to_dict(self):
        return self.__dict__

class LabelMetrics:
    def __init__(self, epoch=None, split=None, label_dict=None):

        if label_dict is None:
            self.epoch = epoch
            self.split = split
            self.crop = Metrics(epoch=epoch, split=split, label='crop')
            self.state = Metrics(epoch=epoch, split=split, label='state')
        else:
            self.epoch = label_dict['epoch']
            self.split = label_dict['split']
            self.crop = Metrics(label_dict['crop'])
            self.state = Metrics(label_dict['state'])
    
    def assign_epoch(self, epoch):
        self.epoch = epoch
        self.crop.epoch = epoch
        self.state.epoch = epoch
        
    def get_formatted_dict(self):
        formatted_dict = {}
        formatted_dict.update(self.crop.get_formatted_dict())
        formatted_dict.update(self.state.get_formatted_dict())
        return formatted_dict
    
    # Function that takes a new LabelMetrics objects and returns the change percentage for each metric
    def change_percentage(self, new_metrics, label: str, metric: str) -> float:
        """ Calculate the percentage change between two metrics within a label

        Args:
            new_metrics (_type_): _description_
            label (str): _description_
            metric (str): _description_

        Returns:
            float: _description_
        """
        return getattr(self, label).change_percentage(getattr(new_metrics, label), metric)
    
    def to_dict(self):
        return {
            'epoch': self.epoch,
            'split': self.split,
            'crop': self.crop.to_dict(),
            'state': self.state.to_dict()
        }


class PerformanceMetrics():
    def __init__(self, epoch=None, performance_dict=None):
        
        if performance_dict is None:
            self.epoch = epoch
            self.completed_datetime_utc = datetime.now(ZoneInfo('UTC'))
            self.completed_datetime_sydney = datetime.now(ZoneInfo('Australia/Sydney'))
            self.train = LabelMetrics(epoch=epoch, split='train')
            self.val = LabelMetrics(epoch=epoch, split='val')
            self.test = LabelMetrics(epoch=epoch, split='test')
        else:
            self.epoch = performance_dict['epoch']
            self.completed_datetime_utc = parser.parse(performance_dict['completed_datetime_utc'])
            self.completed_datetime_sydney = parser.parse(performance_dict['completed_datetime_sydney'])
            self.train = LabelMetrics(performance_dict['train'])
            self.val = LabelMetrics(performance_dict['val'])
            self.test = LabelMetrics(performance_dict['test'])

    
    def assign_epoch(self, epoch):
        self.epoch = epoch
        self.train.assign_epoch(epoch)
        self.val.assign_epoch(epoch)
        self.test.assign_epoch(epoch)
        
    def get_combined_loss(self, split):
        crop_loss = getattr(self, split).crop.loss
        state_loss = getattr(self, split).state.loss
        return crop_loss + state_loss
    
    def get_formatted_metrics_as_dict(self):
        formatted_dict = {}
        formatted_dict.update(self.train.get_formatted_dict())
        formatted_dict.update(self.val.get_formatted_dict())
        formatted_dict.update(self.test.get_formatted_dict())
        return formatted_dict
    
    def to_dict(self):
        return {
            'epoch': self.epoch,
            'completed_datetime_utc': self.completed_datetime_utc.isoformat(),
            'completed_datetime_sydney': self.completed_datetime_sydney.isoformat(),
            'train': self.train.to_dict(),
            'val': self.val.to_dict(),
            'test': self.test.to_dict()
        }
        
        
class ProgressionMetrics():
    def __init__(self, performance_dict=None):
        if performance_dict is None:
            self.past_performance_metrics = []
        else:
            self.past_performance_metrics = [PerformanceMetrics(performance_dict=metric) for metric in performance_dict]
        
    def to_dict(self):
        return [metric.to_dict() for metric in self.past_performance_metrics]
        

        