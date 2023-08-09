from fastapi import FastAPI
from fastapi.responses import RedirectResponse, Response
import wandb
from pydantic import BaseModel
import pickle
import sys

sys.path.append('/Users/chelseatroy/mozilla/mlops-platform-spike-library/bugbug/weights-and-biases/bugbug')
import bugbug
from bugbug import bugzilla

import os

class SwarmOfBugs(BaseModel):
    bug_ids: list

app = FastAPI()

os.system('rm -rf artifacts')
run = wandb.init(project="bugbug-inference-server", job_type="")
artifact = run.use_artifact('mlops-mozilla/model-registry/SpamBug:latest')
artifact.download()
model_directory_name = os.listdir("artifacts")[0]
model_file_name = os.listdir(f"artifacts/{model_directory_name}")[0]

file = open(f"artifacts/{model_directory_name}/{model_file_name}", 'rb')
spambug_model = pickle.load(file)
file.close()

@app.get("/")
async def redirect_home_to_docs():
    """Redirects home endpoint to the interactive documentation provided by FastAPI."""
    response = RedirectResponse(url="/docs")
    return response


@app.post('/spambug_prediction')
async def get_spambug_prediction(swarm: SwarmOfBugs):
    bugs = bugzilla.get(swarm.bug_ids)
    probabilities = spambug_model.classify(list(bugs.values()), True)
    prediction_probabilities = []
    for index, probability_pair in enumerate(probabilities.tolist()):
        bug_proba_dict = {}
        bug_proba_dict['bug_id'] = swarm.bug_ids[index]
        bug_proba_dict['summary'] = bugs[swarm.bug_ids[index]].get('summary')
        bug_proba_dict['creator_detail'] = bugs[swarm.bug_ids[index]].get('creator_detail')
        bug_proba_dict['probability_legitimate_bug'] = probability_pair[0]
        bug_proba_dict['probability_spam_bug'] = probability_pair[1]
        prediction_probabilities.append(bug_proba_dict)

    return {"prediction_probabilities": prediction_probabilities}
