import os
from pathlib import Path
from subprocess import check_output

import six
from mlflow import set_tracking_uri, set_experiment
from google.auth.transport.requests import Request
from google.oauth2 import id_token
import google
import requests

### Source for this code
### https://github.com/ConsciousML/one-click-mlflow

def save_env_var(key, val):
    os.environ[key] = val
    print(f"export {key}={val}")


def get_token():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""
    if "mlflow-log-pusher-key.json" in os.listdir(Path(__file__).parent.parent.parent):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(Path(__file__).parent.parent.parent / "mlflow-log-pusher-key.json")
    try:
        token = _get_token()
    except google.auth.exceptions.DefaultCredentialsError:
        print("You do not seem to have an mlflow-log-pusher service account key locally.")
        prompt = input("Get one ? (Y/n): ") or "y"
        if prompt.lower() == "y":
            print("Fetching SA key...")
            fetch_sa_key()
        token = _get_token()
    return token


def fetch_sa_key():
    check_output([f"gcloud iam service-accounts keys create ./mlflow-log-pusher-key.json --iam-account "
                  f"mlflow-log-pusher@{PROJECT_ID}.iam.gserviceaccount.com"], shell=True)
    save_env_var("GOOGLE_APPLICATION_CREDENTIALS", str(Path(__file__).parent.parent.parent / "mlflow-log-pusher-key.json"))
def _get_token():
    client_id = _get_client_id(tracking_uri)
    open_id_connect_token = id_token.fetch_id_token(Request(), client_id)
    return open_id_connect_token
def _get_client_id(service_uri):
    redirect_response = requests.get(service_uri, allow_redirects=False)
    if redirect_response.status_code != 302:
        print(f"The URI {service_uri} does not seem to be a valid AppEngine endpoint.")
        return None

    redirect_location = redirect_response.headers.get("location")
    if not redirect_location:
        print(f"No redirect location for request to {service_uri}")
        return None

    parsed = six.moves.urllib.parse.urlparse(redirect_location)
    query_string = six.moves.urllib.parse.parse_qs(parsed.query)
    return query_string["client_id"][0]
PROJECT_ID = "moz-fx-dev-ctroy-ml-ops-spikes" # input("Enter your project ID: ")
EXPERIMENT_NAME = "BugBug"
# If mlflow if not deployed on the default app engine service, change it with the url of your service <!-- omit in toc -->
tracking_uri = f"https://mlflow-dot-{PROJECT_ID}.ew.r.appspot.com"
print(f"Trying to set uri to {tracking_uri}")

save_env_var("MLFLOW_TRACKING_TOKEN", get_token())
set_tracking_uri(tracking_uri)
save_env_var("MLFLOW_TRACKING_URI", tracking_uri)
print(f"Tracking uri set to {tracking_uri}")
set_experiment(EXPERIMENT_NAME)

