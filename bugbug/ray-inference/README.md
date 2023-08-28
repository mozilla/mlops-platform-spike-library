Set the thing up.

```shell
python -m venv .venv
.venv/script/activate
pip install -r requirements.txt
```

Run the thing

```shell
serve run template:main_deployment
```

Then:

- http://127.0.0.1:8265 can be used to see the cluster panel
- http://127.0.0.1:8000 to get to the docs
- The endpoints for doing the inference
