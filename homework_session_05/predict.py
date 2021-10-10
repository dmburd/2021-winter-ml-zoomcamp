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

pp = pprint.PrettyPrinter(indent=4)
#pp.pprint(stuff)

from flask import Flask, request, jsonify


#model_fpath = "model1.bin"
model_fpath = "model2.bin"

vectorizer_fpath = "dv.bin"

with open(model_fpath, 'rb') as fd:
    model = pickle.load(fd)

with open(vectorizer_fpath, 'rb') as fd:
    dv = pickle.load(fd)

app = Flask("churn")

@app.route('/predict', methods=["POST"])
def predict():
    customer =request.get_json()
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5
    result = {
        "churn_probability": float(y_pred),
        "churn": bool(churn),
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8006)


# gunicorn --bind 0.0.0.0:8006 predict:app
