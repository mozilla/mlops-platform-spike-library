from typing import Dict

import mlflow_extend.logging
import numpy
import numpy as np
from mlflow.models import ModelSignature, set_signature

import bugbug.trackers.mlflow_config
#import mlflow
import mlflow
from mlflow_extend import mlflow as ml_extend

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

    def log_data_input(self, data:any, name: str):
        if isinstance(data, numpy.ndarray):
            dataset = mlflow.data.from_numpy(data, source=name)
            mlflow.log_input(dataset, context=name)
    def track_metric(self, key: str, data: any):
        mlflow.log_metric(key, data)
    def is_loggable_metric(self, val):
        return type(val) == str or np.isscalar(val)
    def track_all_metrics(self, data: Dict[str, any]):
        def log_dict_recursive(recurse_dict: Dict[str, any], prefix=None):
            prefix_with_separator = f"{prefix}_" if prefix is not None else ""
            for _i, (key, val) in enumerate(recurse_dict.items()):
                next_prefix = f"{prefix_with_separator}{key}"
                if self.is_loggable_metric(val):
                    mlflow.log_metric(f"{next_prefix}", val)
                elif type(val) is dict:
                    log_dict_recursive(val, prefix=f"{next_prefix}")
                else:
                    # We can't log array or some such element - make it an artifact
                    mlflow_extend.logging.log_dict({next_prefix: val}, f"{next_prefix}.json")
        log_dict_recursive(data)
    def _infer_signature(self, input: any, output: any):
        return mlflow.models.infer_signature(input, output)
    def log_scikit_model(self, model, path, input, output):
        sig = self._infer_signature(input, output)
        mlflow.sklearn.log_model(model, artifact_path=path, signature=sig)
    def log_dataframe(self, name, df):
        mlflow.log_table(df, name)
    def log_confusion_matrix(self, confusion_matrix, _labels, name="confusion_matrix"):
        mlflow_extend.logging.log_confusion_matrix(confusion_matrix, path=f"{name}.png")

    def log_artifact(self, filename, name=None):
        mlflow.log_artifact(filename, name)