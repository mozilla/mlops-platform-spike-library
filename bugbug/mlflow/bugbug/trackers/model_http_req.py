import json
from random import random

import requests
import random
# The API endpoint
url = "http://127.0.0.1:5000/invocations"
data_arr = [random.uniform(0, 1) for a in range(26467)]
data = json.dumps(data_arr)
response = requests.post(url, json={"inputs": [data_arr]})
print(response.text)
