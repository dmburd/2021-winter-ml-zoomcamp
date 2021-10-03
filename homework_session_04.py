import sys
import os
import os.path
from os.path import join as pjn
import shutil
from sys import exit as e
from numpy.core import numeric
import q
import pprint

from sklearn.utils.multiclass import type_of_target
pp = pprint.PrettyPrinter(indent=4)
#pp.pprint(stuff)

import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


# Session #4 Homework
# https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/04-evaluation/homework.md

df = pd.read_csv('CreditScoring.csv')
df.columns = df.columns.str.lower()

status_values = {
    1: 'ok',
    2: 'default',
    0: 'unk'
}

df.status = df.status.map(status_values)


home_values = {
    1: 'rent',
    2: 'owner',
    3: 'private',
    4: 'ignore',
    5: 'parents',
    6: 'other',
    0: 'unk'
}

df.home = df.home.map(home_values)

marital_values = {
    1: 'single',
    2: 'married',
    3: 'widow',
    4: 'separated',
    5: 'divorced',
    0: 'unk'
}

df.marital = df.marital.map(marital_values)

records_values = {
    1: 'no',
    2: 'yes',
    0: 'unk'
}

df.records = df.records.map(records_values)

job_values = {
    1: 'fixed',
    2: 'partime',
    3: 'freelance',
    4: 'others',
    0: 'unk'
}

df.job = df.job.map(job_values)

for c in ['income', 'assets', 'debt']:
    df[c] = df[c].replace(to_replace=99999999, value=0)

df = df[df.status != 'unk'].reset_index(drop=True)

df['default'] = (df.status == 'default').astype(int)
del df['status']


dt = df.dtypes.to_dict()
assert set(dt.values()) == set([np.dtype('int64'), np.dtype('O')])
numerical = [k for k, v in dt.items() if v == np.dtype('int64')]


answers = {}
#--------------------------------
seed = 1
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=seed)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=seed)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

target = 'default'

y_full_train = df_full_train[target].values
y_train = df_train[target].values
y_val = df_val[target].values
y_test = df_test[target].values

del df_val[target]
del df_test[target]

roc_auc_di = {}
for feat in numerical:
    roc_auc = roc_auc_score(df_train[target], df_train[feat])
    if roc_auc < 0.5:
        roc_auc = roc_auc_score(df_train[target], -df_train[feat])
    print('\t', feat, roc_auc)
    if feat != target:
        roc_auc_di[feat] = roc_auc

highest_auc_feat = max(roc_auc_di.items(), key=lambda p: p[1])
answers[1] = highest_auc_feat[0]
print('=> Q1:', answers[1])

del df_train[target]


#--------------------------------
feats = ['seniority', 'income', 'assets', 'records', 'job', 'home']

def train_one_model(df_train, y_train, C=1.0):
    dicts = df_train[feats].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
    model.fit(X_train, y_train)
    return dv, model


def predict_one_model(df, dv, model):
    dicts = df[feats].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


dv, model = train_one_model(df_train, y_train)
val_dict = df_val[feats].to_dict(orient='records')
X_val = dv.transform(val_dict)

y_pred = model.predict_proba(X_val)[:, 1]
score = roc_auc_score(y_val, y_pred)
answers[2] = round(score, 3)
print('=> Q2:', answers[2])


#--------------------------------
scores = []

thresholds = np.linspace(0, 1, 101)

for t in thresholds:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)
    
    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)

    tp = (predict_positive & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()

    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()
    
    scores.append((t, tp, fp, fn, tn))


precision_vals = []
recall_vals = []

for scores_tuple in scores:
    t, tp, fp, fn, tn = scores_tuple
    precision_vals.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    recall_vals.append(tp / (tp + fn))

plt.close()
plt.plot(thresholds, precision_vals, label='precision')
plt.plot(thresholds, recall_vals, label='recall')
plt.legend()
plt.savefig("Q3.png")

diff = np.sign(np.array(recall_vals) - np.array(precision_vals)).astype(int)
assert diff[0] == +1
change_sign = [diff[i + 1] - diff[i] for i in range(diff.shape[0] - 1)]
assert change_sign.count(-2) == 1
# ^ exactly one change from +1 to -1 is expected
idx = change_sign.index(-2)
answers[3] = thresholds[idx]
print('=> Q3:', answers[3])


#--------------------------------
F1_vals = []
for scores_tuple in scores:
    t, tp, fp, fn, tn = scores_tuple
    precision = (tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    recall = tp / (tp + fn)
    p_r_sum = precision + recall
    F1_vals.append(2 * precision * recall / p_r_sum if p_r_sum != 0 else 0)

plt.close()
plt.plot(thresholds, F1_vals, label='F1')
plt.legend()
plt.savefig("Q4.png")

idx = np.argmax(np.array(F1_vals))
answers[4] = thresholds[idx]
print('=> Q4:', answers[4])


#--------------------------------
kfold = KFold(n_splits=5, shuffle=True, random_state=1)
scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train_crossval = df_full_train.iloc[train_idx]
    df_val_crossval = df_full_train.iloc[val_idx]

    y_train_crossval = y_full_train[train_idx]
    y_val_crossval = y_full_train[val_idx]

    dv, model = train_one_model(df_train_crossval, y_train_crossval, C=1.0)
    y_pred_crossval = predict_one_model(df_val_crossval, dv, model)

    auc_val = roc_auc_score(y_val_crossval, y_pred_crossval)
    scores.append(auc_val)

std_val = np.std(scores)
answers[5] = round(std_val, 3)
print('=> Q5:', answers[5])


#--------------------------------
C_to_auc_val = {}

for C in [0.01, 0.1, 1, 10]:
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    scores = []

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train_crossval = df_full_train.iloc[train_idx]
        df_val_crossval = df_full_train.iloc[val_idx]

        y_train_crossval = y_full_train[train_idx]
        y_val_crossval = y_full_train[val_idx]

        dv, model = train_one_model(df_train_crossval, y_train_crossval, C=C)
        y_pred_crossval = predict_one_model(df_val_crossval, dv, model)

        auc_val = roc_auc_score(y_val_crossval, y_pred_crossval)
        scores.append(auc_val)
    
    scores = np.array(scores)
    C_to_auc_val[C] = (round(scores.mean(), 3), round(scores.std(), 3))

pp.pprint(C_to_auc_val)
C_for_best_mean_score = max(C_to_auc_val.items(), key=lambda p: p[1][0])
answers[6] = C_for_best_mean_score[0]
print('=> Q6:', answers[6])
