This is a demo project based Docker example at https://github.com/mlflow/mlflow/tree/master/examples/docker
It is adapted for GCS


To Run
gcloud auth login
gcloud iam service-accounts keys create ./mlflow-log-pusher-key.json --iam-account mlflow-log-pusher@moz-fx-dev-ctroy-ml-ops-spikesiam.gserviceaccount.com
MLFLOW_EXPERIMENT_ID=2 # required if you don't want 'Default' experiment
GOOGLE_APPLICATION_CREDENTIALS=mlflow-log-pusher-key.json
python mlflow_config.py > env_vars
source env_vars
sudo docker build . -t us-central1-docker.pkg.dev/moz-fx-dev-ctroy-ml-ops-spikes/bugbug-training-runs-mlflow/example --platform linux/amd64
mlflow run . -P alpha=0.5 --build-image

to run on kubenetes (Working!!):
mlflow run . -P alpha=0.5 --build-image --backend kubernetes --backend-config kubernetes_config.json  --build-image

Note the 'build-image' arg is not in the documentation for this example

Env vars used :
MLFLOW_TRACKING_URI=https://mlflow-dot-moz-fx-dev-ctroy-ml-ops-spikes.ew.r.appspot.com
MLFLOW_TRACKING_TOKEN=[can be set in code as well]
