from abc import abstractmethod
from typing import Dict


class TrackingProvider:

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
