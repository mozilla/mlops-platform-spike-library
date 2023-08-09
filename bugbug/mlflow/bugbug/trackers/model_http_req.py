import json
from random import random
import requests
import random
# The API endpoint
url = "http://127.0.0.1:5000/invocations"

# Request when serving the Xgboost model only
#data_arr = [random.uniform(0, 1) for a in range(26467)]
#response = requests.post(url, json={"inputs": [data]})
#print(response.text)

"""
Request for the spam model
"""

with open("../../tests/fixtures/bugs.json") as f:
    example_bugs = [json.loads(line) for line in f]
for bug in example_bugs:
    bug["filed_via"] = "bugzilla"
prediction_input = [json.dumps(e) for e in example_bugs[:3]]
response = requests.post(url, json={"inputs": prediction_input})
print(response.text)
"""
results -- looks like not spam!
{"predictions": {"probs": [[0.9995030164718628, 0.0004970002337358892], [0.9981616139411926, 0.001838397467508912], [0.9976068735122681, 0.002393108094111085]], "indexes": [0, 0, 0], "suggestions": [0, 0, 0]}}
"""