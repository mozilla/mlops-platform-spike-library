from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from ray import serve

app = FastAPI()


class InferenceResponse(BaseModel):
    result: float
    score: float


@serve.deployment(ray_actor_options={"num_gpus": 0}, num_replicas=1)
class ComplexModel:
    def __init__(self):
        # TODO: any complex initialization here.
        print("Not exactly what you were hoping for.")

    async def __call__(
        self, feature: float
    ) -> InferenceResponse:
        return InferenceResponse(
            # Some fancy AI at play here!
            result=feature * 4.0,
            score=1.0
        )

@serve.deployment()
@serve.ingress(app)
class MainDeployment:
    def __init__(self, some_heavy_model):
        self._some_heavy_model = some_heavy_model

    @app.get("/")
    async def docs_redirect(self):
        return RedirectResponse("/docs")

    @app.post("/inference", response_model=InferenceResponse)
    async def infer_simple(
        self,
        feature: float,
    ) -> InferenceResponse:
        return InferenceResponse(
            # Some fancy AI at play here!
            result=feature * 2.0,
            score=feature / 2.0
        )

    @app.post(
        "/inference_complex",
        response_model=InferenceResponse
    )
    async def infer_complex(
        self,
        feature: float,
    ) -> InferenceResponse:
        ref = await self._some_heavy_model.remote(feature)
        result: InferenceResponse = await ref
        return result


complex_model = ComplexModel.bind()
main_deployment = MainDeployment.bind(
    complex_model
)
