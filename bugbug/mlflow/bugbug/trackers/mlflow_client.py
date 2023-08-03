from mlflow import MlflowClient
from mlflow.entities import ViewType

from bugbug.trackers.mlflow_config import tracking_uri

run = MlflowClient(registry_uri="gs://oneclick-mlflow-store-f8bf20e5c6", tracking_uri=tracking_uri).search_runs(
    experiment_ids="2",
    filter_string="",
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=1,
    order_by=["metrics.test_accuracy_mean DESC"],
)[1]
print(run)