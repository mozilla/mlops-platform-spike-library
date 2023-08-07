from fastapi import FastAPI
from fastapi.responses import RedirectResponse, Response
import wandb
from pydantic import BaseModel

import os
from bugbug.utils import (
    zstd_decompress,
)

class Input(BaseModel):
    features: list

app = FastAPI()

os.system('rm -rf artifacts')
run = wandb.init(project="bugbug-inference-server", job_type="")
artifact = run.use_artifact('mlops-mozilla/model-registry/SpamBug:latest')
artifact.download()
model_directory_name = os.listdir("artifacts")[0]
model_file_name = os.listdir(model_directory_name)[0]
model = zstd_decompress(model_file_name)
print(type(model))

@app.get("/")
async def redirect_home_to_docs():
    """Redirects home endpoint to the interactive documentation provided by FastAPI."""
    response = RedirectResponse(url="/docs")
    return response

@app.post('/spambug_prediction')
async def get_spambug_prediction(input: Input):
   return input
