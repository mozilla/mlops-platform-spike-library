from typing import Dict
import bugbug.trackers.mlflow_config
import mlflow

from bugbug.trackers.tracking_provider import TrackingProvider
class MLFlowTracker(TrackingProvider):
    def start_run(self, name=None):
        mlflow.start_run(run_name=name)

    def set_tag(self, key: str, value: any):
        mlflow.set_tag(key, value)

    def end_run(self):
        mlflow.end_run()

    def track_model_name(self, model_name: str):
        self.set_tag("name", model_name)

    def track_param(self, key: str, data: any):
        mlflow.log_param("bar", 42)
        pass

    def track_metric(self, key: str, data: any):
        mlflow.log_metric(key, data)
        pass

    def track_all_metrics(self, data: Dict[str, any]):
        pass
