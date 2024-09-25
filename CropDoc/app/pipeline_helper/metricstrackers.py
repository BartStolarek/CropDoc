from datetime import datetime
from zoneinfo import ZoneInfo
from dateutil import parser
from loguru import logger


class Metrics:
    def __init__(self, epoch=None, split=None, label=None, metric_dict=None):
        if metric_dict is None:
            if epoch is None or (isinstance(epoch, int) and epoch >= 0):
                self.epoch = epoch
            else:
                raise ValueError("Epoch must be None, positive integer or zero")
            if split is None:
                raise ValueError("Split cannot be None")
            self.split = split
            if label is None:
                raise ValueError("Label cannot be None")
            self.label = label
            self.accuracy = 0
            self.loss = 1e10
            self.precision = 0
            self.recall = 0
            self.f1_score = 0
            self.correctly_classified = 0
            self.total_classifications = 0
            self.confusion_matrix = []
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

    
    def __str__(self):
        return f"METRICS - Epoch ({self.epoch}), Split ({self.split}), Label ({self.label})\n" + \
                f"\t\t\tAccuracy: {self.accuracy:.3f}\n" + \
                f"\t\t\tLoss: {self.loss:.3f}\n" + \
                f"\t\t\tPrecision: {self.precision:.3f}\n" + \
                f"\t\t\tRecall: {self.recall:.3f}\n" + \
                f"\t\t\tF1 Score: {self.f1_score:.3f}\n" + \
                f"\t\t\tCorrectly Classified: {self.correctly_classified}\n" + \
                f"\t\t\tTotal Classifications: {self.total_classifications}\n" + \
                f"\t\t\tConfusion Matrix Length: {len(self.confusion_matrix)}\n"
      
    def get_formatted_dict(self):
        try:
            if self.split == 'train':
                split_letter = 'T'
            elif self.split == 'val':
                split_letter = 'V'
            elif self.split == 'test':
                split_letter = 'TE'

            label_letter = self.label[0].upper()
            return {
                f"{split_letter}{label_letter}A": self.format_accuracy(self.accuracy),
                f"{split_letter}{label_letter}L": self.format_loss(self.loss),
            }
        except TypeError as e:
            raise TypeError(e)
        
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
    
    def to_dict(self):
        return self.__dict__

class SplitMetrics:
    def __init__(self, epoch=None, split=None, split_dict=None):

        if split_dict is None:
            if epoch is None or (isinstance(epoch, int) and epoch >= 0):
                self.epoch = epoch
            else:
                raise ValueError("Epoch must be None, positive integer or zero")
            self.split = split
            self.crop = Metrics(epoch=epoch, split=split, label='crop')
            self.state = Metrics(epoch=epoch, split=split, label='state')
        else:
            self.epoch = split_dict['epoch']
            self.split = split_dict['split']
            self.crop = Metrics(metric_dict=split_dict['crop'])
            self.state = Metrics(metric_dict=split_dict['state'])
    
    def __str__(self):
        return f"SPLIT METRICS - Epoch ({self.epoch}), Split ({self.split})\n" + \
               f"\t\t{self.crop}" + \
               f"\t\t{self.state}"
                
    
    def assign_split(self, split):
        self.split = split
        self.crop.split = split
        self.state.split = split
        
    def assign_label(self, label):
        self.label = label
        self.crop.label = label
        self.state.label = label
    
    def assign_epoch(self, epoch):
        self.epoch = epoch
        self.crop.epoch = epoch
        self.state.epoch = epoch
        
    def get_formatted_dict(self):
        formatted_dict = {}
        formatted_dict.update(self.crop.get_formatted_dict())
        formatted_dict.update(self.state.get_formatted_dict())
        return formatted_dict
    
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
            if epoch is None or (isinstance(epoch, int) and epoch >= 0):
                self.epoch = epoch
            else:
                raise ValueError("Epoch must be None, positive integer or zero")
            self.completed_datetime_utc = datetime.now(ZoneInfo('UTC'))
            self.completed_datetime_sydney = datetime.now(ZoneInfo('Australia/Sydney'))
            self.train = SplitMetrics(epoch=epoch, split='train')
            self.val = SplitMetrics(epoch=epoch, split='val')
            self.test = SplitMetrics(epoch=epoch, split='test')
        else:
            self.epoch = performance_dict['epoch']
            if not (self.epoch is None or (isinstance(self.epoch, int) and self.epoch >= 0)):
                raise ValueError(f"Epoch must be None, positive integer or zero - {self.epoch}")
            self.completed_datetime_utc = parser.parse(performance_dict['completed_datetime_utc'])
            self.completed_datetime_sydney = parser.parse(performance_dict['completed_datetime_sydney'])
            self.train = SplitMetrics(split_dict=performance_dict['train'], split='train')
            self.val = SplitMetrics(split_dict=performance_dict['val'], split='val')
            self.test = SplitMetrics(split_dict=performance_dict['test'], split='test')

    def __str__(self):
        return f"\nPERFORMANCE METRICS - Epoch ({self.epoch})\n" + \
               f"\t{self.train}" + \
               f"\t{self.val}" + \
               f"\t{self.test}"
               
    
    def assign_epoch(self, epoch):
        if isinstance(epoch, dict):
            raise ValueError(f"Epoch cannot be a dictionary - {epoch}")
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
        for key, value in self.train.get_formatted_dict().items():
            formatted_dict[key] = value
        for key, value in self.val.get_formatted_dict().items():
            formatted_dict[key] = value
        for key, value in self.test.get_formatted_dict().items():
            formatted_dict[key] = value
            
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
    def __init__(self, progression_list=None):
        if progression_list is None:
            self.past_performance_metrics = []
        else:
            self.past_performance_metrics = []
            for metric in progression_list:
                if isinstance(metric, dict):
                    self.past_performance_metrics.append(PerformanceMetrics(performance_dict=metric))
                elif isinstance(metric, PerformanceMetrics):
                    self.past_performance_metrics.append(metric)
                else:
                    raise TypeError("Can only append dictionaries or PerformanceMetrics objects")
    
    def remove_duplicate_performance_metrics(self):
        epochs = []
        new_list = []
        for metric in self.past_performance_metrics:
            if metric.epoch not in epochs:
                epochs.append(metric.epoch)
                new_list.append(metric)
        self.past_performance_metrics = new_list
        
    def __iter__(self):
        return iter(self.past_performance_metrics)
    
    def __len__(self):
        return len(self.past_performance_metrics)
    
    def __getitem__(self, idx):
        return self.past_performance_metrics[idx]
    
    def __setitem__(self, idx, value):
        self.past_performance_metrics[idx] = value
    
    def append(self, performance_metric):
        if isinstance(performance_metric, PerformanceMetrics):
            self.past_performance_metrics.append(performance_metric)
        else:
            raise TypeError("Can only append PerformanceMetrics objects")
    
    def to_list(self):
        result = []
        for metric in self.past_performance_metrics:
            try:
                result.append(metric.to_dict())
            except Exception as e:
                print(f"Error converting metric to dict at index {i}: {e}")
                print(f"Metric: {metric}")
        return result

