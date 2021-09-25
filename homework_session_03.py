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

from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score, mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, Ridge


# Session #3 Homework
# https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/03-classification/homework.md

answers = {}

df = pd.read_csv("AB_NYC_2019.csv")

numerical = [
    'latitude',
    'longitude',
    'price',
    'minimum_nights',
    'number_of_reviews',
    'reviews_per_month',
    'calculated_host_listings_count',
    'availability_365'
]
categorical = [
    'neighbourhood_group',
    'room_type',
]
required_columns = categorical + numerical
df = df[required_columns]
df = df.fillna(0)

#--------------------------------
ng = df.neighbourhood_group
answers[1] = ng.mode().values[0]
print('=> Q1:', answers[1])


#--------------------------------
seed = 42
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=seed)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=seed)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.price.values
y_val = df_val.price.values
y_test = df_test.price.values

del df_train['price']
del df_val['price']
del df_test['price']


#--------------------------------
corr = df_train.corr()
corr_np = corr.values
tmp1 = corr_np * (1 - np.eye(corr_np.shape[0]))
tmp2 = np.abs(tmp1)
inds = np.unravel_index(np.argmax(tmp2, axis=None), tmp2.shape)
max_corr_features = [corr.columns[i] for i in inds]
answers[2] = max_corr_features
print('=> Q2:', answers[2])

#--------------------------------
above_average_train = (y_train >= 152).astype(int)
above_average_val = (y_val >= 152).astype(int)

mi_dict = {c: mutual_info_score(df_train[c], above_average_train) for c in categorical}
max_mi_entry = max(mi_dict.items(), key=lambda kv: kv[1])
max_mi_entry = max_mi_entry[0], round(max_mi_entry[1], 2)
answers[3] = max_mi_entry
print('=> Q3:', answers[3])


#--------------------------------
def train_one_model(df_train, above_average_train, df_val, above_average_val):
    dv = DictVectorizer(sparse=False)
    train_dict = df_train.to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)

    val_dict = df_val.to_dict(orient='records')
    X_val = dv.transform(val_dict)

    model = LogisticRegression(solver='lbfgs', C=1.0, random_state=42)
    model.fit(X_train, above_average_train)
    above_average_pred = model.predict_proba(X_val)[:, 1]
    binary_decision = (above_average_pred >= 0.5)
    acc = (binary_decision == above_average_val).mean()
    return acc

acc_all_features = train_one_model(df_train, above_average_train, df_val, above_average_val)
answers[4] = round(acc_all_features, 2)
print('=> Q4:', answers[4])


#--------------------------------
acc_dict = {}
# ^ maps excluded feature name to the accuracy value of the model trained after excluding that feature
all_features = set(categorical) | set(numerical) - set(['price',])
for one_feature in all_features:
    remaining_features = all_features - set([one_feature,])
    acc_dict[one_feature] = train_one_model(
        df_train[remaining_features],
        above_average_train,
        df_val[remaining_features],
        above_average_val
    )

abs_diff_acc_dict = {k: abs(v - acc_all_features) for k, v in acc_dict.items()}
selected_features_list = [
    'neighbourhood_group',
    'room_type',
    'number_of_reviews',
    'reviews_per_month',
]
selected_abs_diff_acc_dict = {k: v for k, v in abs_diff_acc_dict.items() if k in selected_features_list}
min_entry = min(selected_abs_diff_acc_dict.items(), key=lambda kv: kv[1])
answers[5] = min_entry[0]
print('=> Q5:', answers[5])


#--------------------------------
y_train = np.log1p(y_train)
y_val = np.log1p(y_val)
y_test = np.log1p(y_test)

alpha_list = [0, 0.01, 0.1, 1, 10]
rmse_dict = {}

for alpha in alpha_list:
    dv = DictVectorizer(sparse=False)
    train_dict = df_train.to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)
    val_dict = df_val.to_dict(orient='records')
    X_val = dv.transform(val_dict)
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred)
    rmse_dict[alpha] = round(rmse, 3)

pp.pprint(rmse_dict)
min_entry = min(rmse_dict.items(), key=lambda kv: kv[1])
answers[6] = min_entry[0]
print('=> Q6:', answers[6])
