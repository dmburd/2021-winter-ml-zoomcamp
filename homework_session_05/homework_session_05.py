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


# Session #5 Homework
# https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/05-deployment/homework.md

output = subprocess.check_output(['pip', 'install', 'pipenv',])
decoded = output.decode(encoding='utf-8')
#print(decoded)

output = subprocess.check_output(['pipenv', '--version',])
decoded = output.decode(encoding='utf-8')
i = decoded.find('version')
assert i != -1
ver = decoded[(i + len('version')):].strip()
print('=> Q1:', ver)


output = subprocess.check_output(['pipenv', 'install', 'scikit-learn==1.0',])
with open("Pipfile.lock", 'r') as fd:
    contents = json.load(fd)

first_hash = contents['default']['scikit-learn']['hashes'][0]
print('=> Q2:', first_hash)


with open("model1.bin", 'rb') as fd:
    model = pickle.load(fd)

with open("dv.bin", 'rb') as fd:
    dv = pickle.load(fd)

customer = {"contract": "two_year", "tenure": 12, "monthlycharges": 19.7}
X = dv.transform([customer,])
score = model.predict_proba(X)[0, 1]
print('=> Q3:', score)


agrigorev_zoomcamp_model_image_id = "f0f43f7bc6e0"
print('=> Q5:', agrigorev_zoomcamp_model_image_id)

