from abc import abstractmethod
from enum import Enum, auto
from typing import Dict

class ModelType(Enum):
     SKLearn = auto()

class TrackingProvider:

    def __init__(self, model_type : ModelType = None):
        self.model_type = model_type

    def log_scikit_model(self, model, name, input, output):
        pass

    def start_run(self, name=None):
        pass

    def set_tag(self, key: str, value: any):
        """
        Call this after the run has started
        """
        pass

    def end_run(self):
        pass

    def track_param(self, key: str, data: any):
        """
        Track a training Param
        """
        pass

    """
    Abstract class for tracing providers such as MLFlow, WandB, etc.
    """
    @abstractmethod
    def track_metric(self, key: str, data: any):
        """
        Tracks a single metric on a tracking server
        """
        pass

    @abstractmethod
    def track_all_metrics(self, data: Dict[str, any]):
        """
        Track multiple metrics on a tracking server
        Each key is tracked separately
        """
        pass

    @abstractmethod
    def log_scikit_model(self, model, path, input, output):
       pass

    @abstractmethod
    def log_pyfunc_model(self, model, path, input, output):
        pass
    @abstractmethod
    def log_artifact(self, filename, name=None):
        pass

    @abstractmethod
    def log_data_input(self, data:any, name: str):
        pass
    @abstractmethod
    def log_dataframe(self, name, df):
        pass

    @abstractmethod
    def log_confusion_matrix(self, confusion_matrix, labels, name=None):
        pass
