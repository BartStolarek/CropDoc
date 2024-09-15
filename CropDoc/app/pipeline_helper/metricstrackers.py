from datetime import datetime
from zoneinfo import ZoneInfo
from dateutil import parser


class Metrics:
    def __init__(self, epoch=None, split=None, label=None, metric_dict=None):
        if metric_dict is None:
            self.epoch = epoch
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
        return f"Epoch: {self.epoch}, Split: {self.split}, Label: {self.label}, Accuracy: {self.accuracy}, Loss: {self.loss}, Precision: {self.precision}, Recall: {self.recall}, F1 Score: {self.f1_score}, Correctly Classified: {self.correctly_classified}, Total Classifications: {self.total_classifications}"
      
    def get_formatted_dict(self):
        try:
            if self.split is 'train':
                split_letter = 'T'
            elif self.split is 'val':
                split_letter = 'V'
            elif self.split is 'test':
                split_letter = 'TE'

            label_letter = self.label[0].upper()
            return {
                f"{split_letter}{label_letter}A": self.format_accuracy(self.accuracy),
                f"{split_letter}{label_letter}L": self.format_loss(self.loss),
            }
        except TypeError as e:
            print(f"Error in get_formatted_dict: {self}")
            print(f"self.accuracy: {self.accuracy}, self.loss: {self.loss}")
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
    
    def __str__(self):
        return f"Epoch: {self.epoch}, Split: {self.split}, \nCrop: {self.crop}, \nState: {self.state}"
    
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
            self.train = LabelMetrics(performance_dict['train'], split='train')
            self.val = LabelMetrics(performance_dict['val'], split='val')
            self.test = LabelMetrics(performance_dict['test'], split='test')

    def __str__(self):
        return f"Epoch: {self.epoch}, \nTrain: {self.train}, \nVal: {self.val}, \nTest: {self.test}"
    
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
    def __init__(self, performance_dict=None):
        if performance_dict is None:
            self.past_performance_metrics = []
        else:
            self.past_performance_metrics = [PerformanceMetrics(performance_dict=metric) for metric in performance_dict]
        
    def to_dict(self):
        result = []
        for i, metric in enumerate(self.past_performance_metrics):
            try:
                result.append(metric.to_dict())
            except Exception as e:
                print(f"Error converting metric to dict at index {i}: {e}")
                print(f"Metric: {metric}")
        return result
        