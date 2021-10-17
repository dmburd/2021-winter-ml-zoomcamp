# run the following commands before running this script:
# export XGBOOST_ETA=0.3; python homework_session_06_xgboost.py > xgboost_eta_${XGBOOST_ETA}.txt
# export XGBOOST_ETA=0.1; python homework_session_06_xgboost.py > xgboost_eta_${XGBOOST_ETA}.txt
# export XGBOOST_ETA=0.01; python homework_session_06_xgboost.py > xgboost_eta_${XGBOOST_ETA}.txt

import sys
import os
import os.path
from os.path import join as pjn
import shutil
from sys import exit as e
from typing import OrderedDict
import q
import pprint

pp = pprint.PrettyPrinter(indent=4)
#pp.pprint(stuff)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import export_text, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Session #6 Homework
# https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/06-trees/homework.md

columns = [
    'neighbourhood_group', 'room_type', 'latitude', 'longitude',
    'minimum_nights', 'number_of_reviews','reviews_per_month',
    'calculated_host_listings_count', 'availability_365',
    'price'
]

df = pd.read_csv('AB_NYC_2019.csv', usecols=columns)
df.reviews_per_month = df.reviews_per_month.fillna(0)


#--------------------------------
seed = 1
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=seed)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=seed)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.price.values
y_val = df_val.price.values
y_test = df_test.price.values

y_train = np.log1p(y_train)
y_val = np.log1p(y_val)
y_test = np.log1p(y_test)

del df_train['price']
del df_val['price']
del df_test['price']


#--------------------------------
def train_one_model(df_train, y_train, df_val, y_val, model_type, random_state, n_estimators, max_depth):
    assert model_type in ('DecisionTreeRegressor', 'RandomForestRegressor')

    dv = DictVectorizer(sparse=False)
    train_dict = df_train.to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)

    val_dict = df_val.to_dict(orient='records')
    X_val = dv.transform(val_dict)

    if model_type == 'RandomForestRegressor':
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
    else:
        model = DecisionTreeRegressor(
            max_depth=max_depth,
            random_state=random_state
        )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred)
    return model, dv, rmse


model, dv, rmse = train_one_model(
    df_train, y_train,
    df_val, y_val,
    'DecisionTreeRegressor',
    random_state=seed,
    n_estimators=0,
    max_depth=1
)
feature_names = list(dv.get_feature_names_out())
tree_txt = export_text(model, feature_names=feature_names)
print(tree_txt)
first_line = tree_txt.split('\n')[0]
first_line = first_line.strip(' |-')
feat_name = first_line.split('=')[0]
assert feat_name in columns
print('=> Q1:', feat_name)


#--------------------------------
model, dv, rmse = train_one_model(
    df_train, y_train,
    df_val, y_val,
    'RandomForestRegressor',
    n_estimators=10,
    max_depth=None, # default
    random_state=seed,
)
print('=> Q2:', round(rmse, 3))


#--------------------------------
rmse_dict = OrderedDict()

for ne in range(10, 210, 10):
    model, dv, rmse = train_one_model(
        df_train, y_train,
        df_val, y_val,
        'RandomForestRegressor',
        n_estimators=ne,
        max_depth=None, # default
        random_state=seed,
    )
    rmse_dict[ne] = round(rmse, 6)

#pp.pprint(rmse_dict)

best_pair = min(rmse_dict.items(), key=lambda pair: pair[1])
best_n = best_pair[0]
plt.close()
fig, ax = plt.subplots()
ax.plot(rmse_dict.keys(), rmse_dict.values())
ax.set_xlabel('n_estimators', fontsize=15)
ax.set_ylabel('rmse', fontsize=15)
ax.grid(True)
fig.tight_layout()
plt.savefig("Q3.png")
print('=> Q3:', 'see Q3.png')


#--------------------------------
rmse_dict = {}

for md in [10, 15, 20, 25]:
    for ne in range(10, 210, 10):
        model, dv, rmse = train_one_model(
            df_train, y_train,
            df_val, y_val,
            'RandomForestRegressor',
            n_estimators=ne,
            max_depth=md,
            random_state=seed,
        )
        rmse_dict[(md, ne)] = round(rmse, 6)

#pp.pprint(rmse_dict)

best_pair = min(rmse_dict.items(), key=lambda pair: pair[1])
best_md = best_pair[0][0]
print('=> Q4:', best_md)


#--------------------------------
model, dv, rmse = train_one_model(
    df_train, y_train,
    df_val, y_val,
    'RandomForestRegressor',
    n_estimators=10,
    max_depth=20,
    random_state=seed,
)

feat_imp = model.feature_importances_
pairs = zip(
    list(dv.get_feature_names_out()),
    list(feat_imp),
)
best_pair = max(pairs, key=lambda pair: pair[1])
most_imp_feat = best_pair[0]
print('=> Q5:', most_imp_feat)


#--------------------------------
def parse_xgb_output(fpath):
    results = []

    with open(fpath, 'r') as fd:
        for line in fd:
            it_line, train_line, val_line = line.split('\t')

            it = int(it_line.strip('[]'))
            train = float(train_line.split(':')[1])
            val = float(val_line.split(':')[1])

            results.append((it, train, val))
    
    columns = ['num_iter', 'train_metric_value', 'val_metric_value']
    df_results = pd.DataFrame(results, columns=columns)
    return df_results
    

eta_list = [0.3, 0.1, 0.01]
df_results_dict = {}
min_rmse_dict = {}

for eta in eta_list:
    df_results_dict[eta] = parse_xgb_output(f"xgboost_eta_{eta}.txt")
    min_rmse_dict[eta] = min(df_results_dict[eta].val_metric_value)

best_pair = min(min_rmse_dict.items(), key=lambda pair: pair[1])
best_eta = best_pair[0]
print('=> Q6:', best_eta)
