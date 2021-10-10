import sys
import os
import os.path
from os.path import join as pjn
import shutil
from sys import exit as e
import q
import pprint
import pickle
import json
import subprocess

pp = pprint.PrettyPrinter(indent=4)
#pp.pprint(stuff)

import numpy as np
import pandas as pd
import requests


url = "http://localhost:8006/predict"

customer = {"contract": "two_year", "tenure": 1, "monthlycharges": 10}
response = requests.post(url, json=customer)
response_dict = response.json()
#print(response_dict)
print('=> Q4:', response_dict['churn_probability'])


customer = {"contract": "two_year", "tenure": 12, "monthlycharges": 10}
response = requests.post(url, json=customer)
response_dict = response.json()
#print(response_dict)
print('=> Q6:', response_dict['churn_probability'])
