from typing import Dict
import bugbug.trackers.mlflow_config
import mlflow

from bugbug.trackers.tracking_provider import TrackingProvider
class MLFlowTracker(TrackingProvider):
    def start_run(self, name=None):
        mlflow.start_run(run_name=name)
        mlflow.sklearn.autolog()
    def set_tag(self, key: str, value: any):
        mlflow.set_tag(key, value)

    def end_run(self):
        mlflow.end_run()

    def track_model_name(self, model_name: str):
        self.set_tag("name", model_name)

    def track_param(self, key: str, data: any):
        mlflow.log_param(key, data)

    def track_metric(self, key: str, data: any):
        mlflow.log_metric(key, data)

    def track_all_metrics(self, data: Dict[str, any]):
        for key, val in enumerate(data):
            if type(val) == str or type(val) == float or type(val) == int:
                mlflow.log_metric(key, val)
