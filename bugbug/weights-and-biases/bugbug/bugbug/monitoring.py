import wandb

class WeightsAndBiasesClient():
    def log_mozilla_artifact(
            cls,
            wandb_run,
            artifact_name="metrics_file",
            artifact_type="data",
            local_filpath="metrics.json"
    ):
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
        artifact.add_file(local_filpath)
        wandb_run.log_artifact(artifact)

    def set_summary_metric(cls, key, value):
        wandb.summary[key] = value

