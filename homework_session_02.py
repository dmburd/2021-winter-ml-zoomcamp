import sys
import os
import os.path
from os.path import join as pjn
import shutil
from sys import exit as e
import q
import pprint
pp = pprint.PrettyPrinter(indent=4)
#pp.pprint(stuff)

import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Session #2 Homework
# https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/02-regression/homework.md

answers = {}

df = pd.read_csv("AB_NYC_2019.csv")

sns.histplot(df.price, bins=50)
plt.savefig('price.png')
# >>> df.price.describe()
# count    48895.000000
# mean       152.720687
# std        240.154170
# min          0.000000
# 25%         69.000000
# 50%        106.000000
# 75%        175.000000
# max      10000.000000
# ^ of course the price distribution has long tail

required_columns = [
    'latitude',
    'longitude',
    'price',
    'minimum_nights',
    'number_of_reviews',
    'reviews_per_month',
    'calculated_host_listings_count',
    'availability_365'
]
df = df[required_columns]


#--------------------------------
nums_of_missing_vals = df.isnull().sum()
nums_of_missing_vals = nums_of_missing_vals[nums_of_missing_vals > 0]
assert len(nums_of_missing_vals) == 1
# ^ only one feature is expected to have missing values
assert nums_of_missing_vals.keys()[0] == 'reviews_per_month'
# ^ and it is 'reviews_per_month'
answers[1] = nums_of_missing_vals.values[0]
print('=> Q1:', answers[1])


#--------------------------------
answers[2] = df.minimum_nights.median()
print('=> Q2:', answers[2])


#--------------------------------
n = len(df)
idx = np.arange(n)
seed = 42
np.random.seed(seed)
np.random.shuffle(idx)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train+n_val]]
df_test = df.iloc[idx[n_train+n_val:]]

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = np.log1p(df_train.price.values)
y_val = np.log1p(df_val.price.values)
y_test = np.log1p(df_test.price.values)

del df_train['price']
del df_val['price']
del df_test['price']


#--------------------------------
def prepare_X(df, value_for_fillna=0):
    df = df.copy()
    df = df.fillna(value_for_fillna)
    X = df.values
    return X


def train_linear_regression_reg(X, y, r=0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])

    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    
    return w_full[0], w_full[1:]


def rmse(y_gt, y_pred):
    se = (y_gt - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)


#--------------------------------
mean_reviews_per_month = df_train.reviews_per_month.mean()
scores = []

for v in (0, mean_reviews_per_month):
    X_train = prepare_X(df_train, value_for_fillna=v)
    w0, w = train_linear_regression_reg(X_train, y_train, r=0)
    X_val = prepare_X(df_val, value_for_fillna=v)
    y_pred = w0 + X_val.dot(w)
    err = rmse(y_val, y_pred)
    scores.append(round(err, 2))
    print(f"    RMSE value for the case of fillna({round(v, 2)}): {scores[-1]}")

answers[3] = (scores[0] == scores[1])
assert answers[3]
print('=> Q3:', 'equally good')


#--------------------------------
r_list = [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]
scores = []

for r in r_list:
    X_train = prepare_X(df_train, value_for_fillna=0)
    w0, w = train_linear_regression_reg(X_train, y_train, r=r)
    X_val = prepare_X(df_val, value_for_fillna=0)
    y_pred = w0 + X_val.dot(w)
    err = rmse(y_val, y_pred)
    scores.append(round(err, 2))
    print(f"    RMSE value for r={r}: {scores[-1]}")

best_rmse = min(scores)
min_best_r = r_list[scores.index(best_rmse)]
answers[4] = min_best_r
print('=> Q4:', answers[4])


#--------------------------------
seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
scores = []

n = len(df)
for s in seed_list:
    idx = np.arange(n)
    np.random.seed(s)
    np.random.shuffle(idx)
    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test

    df_train = df.iloc[idx[:n_train]]
    df_val = df.iloc[idx[n_train:n_train+n_val]]
    df_test = df.iloc[idx[n_train+n_val:]]

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = np.log1p(df_train.price.values)
    y_val = np.log1p(df_val.price.values)
    y_test = np.log1p(df_test.price.values)

    del df_train['price']
    del df_val['price']
    del df_test['price']

    X_train = prepare_X(df_train, value_for_fillna=0)
    w0, w = train_linear_regression_reg(X_train, y_train, r=0)
    X_val = prepare_X(df_val, value_for_fillna=0)
    y_pred = w0 + X_val.dot(w)
    err = rmse(y_val, y_pred)
    scores.append(err)

scores = np.array(scores)
answers[5] = round(np.std(scores), 3)
print('=> Q5:', answers[5])


#--------------------------------
n = len(df)
idx = np.arange(n)
s = 9
np.random.seed(s)
np.random.shuffle(idx)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train+n_val]]
df_test = df.iloc[idx[n_train+n_val:]]

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

df_full_train = pd.concat([df_train, df_val])
df_full_train = df_full_train.reset_index(drop=True)

y_full_train = np.log1p(df_full_train.price.values)
y_test = np.log1p(df_test.price.values)

del df_full_train['price']
del df_test['price']

X_full_train = prepare_X(df_full_train, value_for_fillna=0)
w0, w = train_linear_regression_reg(X_full_train, y_full_train, r=0.001)
X_test = prepare_X(df_test, value_for_fillna=0)
y_pred = w0 + X_test.dot(w)
err = rmse(y_test, y_pred)
answers[6] = round(err, 2)
print('=> Q6:', answers[6])
