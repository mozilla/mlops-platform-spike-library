from typing import Dict

import numpy as np
from mlflow.models import ModelSignature, set_signature

import bugbug.trackers.mlflow_config
#import mlflow
from mlflow_extend import mlflow

from bugbug.trackers.tracking_provider import TrackingProvider, ModelType


class MLFlowTracker(TrackingProvider):
    def start_run(self, name=None, positive_label=None):
        self.name = name
        mlflow.start_run(run_name=name)
    def log_scikit_model(self, model, name, input, output):
        mlflow.sklearn.log_model(model, name, input, output)
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
    def is_loggable_metric(self, val):
        return type(val) == str or np.isscalar(val)
    def track_all_metrics(self, data: Dict[str, any]):
        for _i, (key, val) in enumerate(data.items()):
            if self.is_loggable_metric(val):
                mlflow.log_metric(key, val)
            elif type(val) is dict:
                for _j, (subkey, subval) in enumerate(val.items()):
                    if self.is_loggable_metric(subval):
                        mlflow.log_metric(f"{key}_{subkey}", subval)
                    elif type(subval) is dict:
                        for _j, (subkey2, subval2) in enumerate(subval.items()):
                            if self.is_loggable_metric(subval2):
                                mlflow.log_metric(f"{key}_{subkey}_{subkey2}", subval2)

    def _infer_signature(self, input: any, output: any):
        return mlflow.models.infer_signature(input, output)

    def log_scikit_model(self, model, path, input, output):
        sig = self._infer_signature(input, output)
        mlflow.sklearn.log_model(model, artifact_path=path, signature=sig)

