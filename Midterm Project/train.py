#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#from sklearn.metrics import mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.ensemble import RandomForestClassifier
import pickle

# parameters
max_depth = 10
min_samples_leaf = 10
n_estimators = 80

# output file
output_file = 'project1_model_FR.bin'

# data preparation
df = pd.read_csv('data/ai4i2020.csv')
df.columns = df.columns.str.strip(' [K]').str.strip(' [rpm').str.strip(' [N').str.strip(' [min').str.lower().str.replace(' ', '_')
df['target'] = df.twf + df.hdf + df.pwf + df.osf + df.rnf
df.target = df.target.astype(bool)
df.target = df.target.astype(int)
categorical = ['type']
numerical = ['air_temperature', 'process_temperature', 'rotational_speed', 'torque', 'tool_wear_min']
del df['udi']
del df['product_id']
del df['machine_failure']

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.target.values
y_val = df_val.target.values
y_test = df_test.target.values

y_train_twf = df_train.twf.values
y_val_twf = df_val.twf.values
y_test_twf = df_test.twf.values

y_train_hdf = df_train.hdf.values
y_val_hdf = df_val.hdf.values
y_test_hdf = df_test.hdf.values

y_train_pwf = df_train.pwf.values
y_val_pwf = df_val.pwf.values
y_test_pwf = df_test.pwf.values

y_train_osf = df_train.osf.values
y_val_osf = df_val.osf.values
y_test_osf = df_test.osf.values

y_train_rnf = df_train.rnf.values
y_val_rnf = df_val.rnf.values
y_test_rnf = df_test.rnf.values

del df_train['target']
del df_val['target']
del df_test['target']

del df_train['twf']
del df_val['twf']
del df_test['twf']

del df_train['hdf']
del df_val['hdf']
del df_test['hdf']

del df_train['pwf']
del df_val['pwf']
del df_test['pwf']

del df_train['osf']
del df_val['osf']
del df_test['osf']

del df_train['rnf']
del df_val['rnf']
del df_test['rnf']


# One-hot encoding
dv = DictVectorizer(sparse=False)
train_dict = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val.to_dict(orient='records')
X_val = dv.transform(val_dict)

# Random forest training

rf = RandomForestClassifier(n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf,
                            random_state=1)
rf.fit(X_train, y_train)

# validation

print('doing validation')

y_pred = rf.predict_proba(X_val)[:, 1]
auc_val = roc_auc_score(y_val, y_pred)
print('validation results:')
print(f'auc={auc_val}')

# Test the model on the test dataset
print('doing test on the test dataset')

df_full_train = df_full_train.reset_index(drop=True)
y_full_train = df_full_train.target.values
del df_full_train['target']
dicts_full_train = df_full_train.to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)
dicts_test = df_test.to_dict(orient='records')
X_test = dv.transform(dicts_test)

rf = RandomForestClassifier(n_estimators=80,
                            max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf,
                            random_state=1)
rf.fit(X_full_train, y_full_train)

y_pred_rf_test = rf.predict_proba(X_test)[:, 1]
auc_test = roc_auc_score(y_test, y_pred_rf_test)
print('test results:')
print(f'auc={auc_test}')

# Save file

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, rf), f_out)

print(f'the model is saved to {output_file}')