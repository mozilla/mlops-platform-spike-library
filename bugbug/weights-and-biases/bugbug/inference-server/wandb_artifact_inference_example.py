import os
import sys

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from ray import serve
from pydantic import BaseModel
import wandb
import pickle

sys.path.append('/Users/chelseatroy/mozilla/mlops-platform-spike-library/bugbug/weights-and-biases/bugbug')
from bugbug import bugzilla

app = FastAPI()

class SwarmOfBugs(BaseModel):
    bug_ids: list

class InferenceResponse(BaseModel):
    bug_probabilities: list

@serve.deployment(ray_actor_options={"num_gpus": 0}, num_replicas=1)
class SpamBugModel:
    def __init__(self):
        os.system('rm -rf artifacts')
        run = wandb.init(project="bugbug-inference-server", job_type="")
        artifact = run.use_artifact('mlops-mozilla/model-registry/SpamBug:latest')
        artifact.download()
        model_directory_name = os.listdir("artifacts")[0]
        model_file_name = os.listdir(f"artifacts/{model_directory_name}")[0]

        file = open(f"artifacts/{model_directory_name}/{model_file_name}", 'rb')
        self.model = pickle.load(file)
        file.close()

    async def __call__(
        self, swarm: SwarmOfBugs
    ) -> InferenceResponse:
        bugs = bugzilla.get(swarm.bug_ids)

        probabilities = self.model.classify(list(bugs.values()), True)

        response = InferenceResponse(bug_probabilities=[])

        for index, probability_pair in enumerate(probabilities.tolist()):
            bug_proba_dict = {}
            bug_proba_dict['bug_id'] = swarm.bug_ids[index]
            bug_proba_dict['summary'] = bugs[swarm.bug_ids[index]].get('summary')
            bug_proba_dict['creator_detail'] = bugs[swarm.bug_ids[index]].get('creator_detail')
            bug_proba_dict['probability_legitimate_bug'] = probability_pair[0]
            bug_proba_dict['probability_spam_bug'] = probability_pair[1]
            response.bug_probabilities.append(bug_proba_dict)

        return response

@serve.deployment()
@serve.ingress(app)
class MainDeployment:
    def __init__(self, some_heavy_model):
        self._some_heavy_model = some_heavy_model

    @app.get("/")
    async def docs_redirect(self):
        return RedirectResponse("/docs")

    @app.post(
        "/spambug_prediction",
        response_model=InferenceResponse
    )
    async def infer_complex(
        self,
        swarm: SwarmOfBugs,
    ) -> InferenceResponse:
        ref = await self._some_heavy_model.remote(swarm)
        result: InferenceResponse = await ref
        return result


complex_model = SpamBugModel.bind()
main_deployment = MainDeployment.bind(
    complex_model
)
