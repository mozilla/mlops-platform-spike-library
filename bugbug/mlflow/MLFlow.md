Training with MLFlow

## To Train

gcloud auth login

gcloud iam service-accounts keys create ./mlflow-log-pusher-key.json --iam-account mlflow-log-pusher@moz-fx-dev-ctroy-ml-ops-spikes.iam.gserviceaccount.com
export MLFLOW_TRACKING_URI=https://mlflow-dot-moz-fx-dev-ctroy-ml-ops-spikes.ew.r.appspot.com
MLFLOW_EXPERIMENT_ID=2 # required if you don't want 'Default' experiment
GOOGLE_APPLICATION_CREDENTIALS=mlflow-log-pusher-key.json
python mlflow_config.py > env_vars
source env_vars

If you train locally (i.e. 'python -m scripts.trainer spambug') the training run will be logged

# Docker
Remove any big files from the data directory
rm data/*

### Build the base build
sudo docker build . -f infra/dockerfile_no_module.yml -t us-central1-docker.pkg.dev/moz-fx-dev-ctroy-ml-ops-spikes/bugbug-training-runs-mlflow/bugbug_training_base  --platform linux/amd64

### Run locally
mlflow run . --build-image

## To run on kubenetes

mlflow run . --build-image --backend kubernetes  --backend-config mlflow-kubernetes-config.json

Note the 'build-image' arg is not in all MLFlow documentation and is needed in most cases


