from fastapi import FastAPI
from fastapi.responses import RedirectResponse, Response
import wandb
from pydantic import BaseModel
import pickle

import os

class Bug(BaseModel):
    values: list

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
async def get_spambug_prediction(input: Bug):
    result = spambug_model.classify(input.values, True)

    return {"prediction": result}
