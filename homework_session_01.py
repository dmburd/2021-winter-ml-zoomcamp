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


# Session #1 Homework
# https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/01-intro/homework.md

answers = {
    1: np.__version__,
    2: pd.__version__,
}
for k, v in answers.items():
    print(f"=> Q{k}: {v}")


df = pd.read_csv(
    pjn(os.pardir, "chapter-02-car-price", "data.csv")
)

#--------------------------------
#df[df.Make == 'BMW'].MSRP.describe()
answers[3] = df[df.Make == 'BMW'].MSRP.mean()
print('=> Q3:', round(answers[3], 2))


#--------------------------------
tmp1 = df[df.Year >= 2015]['Engine HP']
answers[4] = tmp1.isnull().sum()
print('=> Q4:', answers[4])


#--------------------------------
print()
print('before fillna():')
mean_hp_before = df['Engine HP'].mean()
print('    num. of nan vals: ', df['Engine HP'].isnull().sum())
print('    average Engine HP:', mean_hp_before)
#
df['Engine HP'] = df['Engine HP'].fillna(mean_hp_before)
#
print('after fillna():')
mean_hp_after = df['Engine HP'].mean()
print('    num. of nan vals: ', df['Engine HP'].isnull().sum())
print('    average Engine HP:', df['Engine HP'].mean())
print()
same = (round(mean_hp_before) == round(mean_hp_after))
answers[5] = ('unchanged' if same else 'changed')
print('=> Q5:', answers[5])


#--------------------------------
df1 = df[df.Make == 'Rolls-Royce'][
    ['Engine HP', 'Engine Cylinders', 'highway MPG']
].drop_duplicates()
X = df1.values
XTX = X.T.dot(X)
inverse_XTX = np.linalg.inv(XTX)
answers[6] = inverse_XTX.sum()
print('=> Q6:', answers[6])


#--------------------------------
y = np.array(
    [1000, 1100, 900, 1200, 1000, 850, 1300]
)
tmp2 = inverse_XTX.dot(X.T)
w = tmp2.dot(y)
answers[7] = w[0]
print('=> Q7:', round(answers[7], 4))

#q.d()
