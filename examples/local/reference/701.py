#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 15:50:54 2019

@author: pablo
"""

import pandas as pd
import numpy as np
import shapefile  # Execute 'pip install pyshp' if this package is not installed in your computer
import datetime
import os  # pip install os
import matplotlib.pyplot as plt

from sklearn import linear_model

# Working directory
cwd = os.getcwd()

# PATH_TO_TRAIN = '/data/speed/sample-data/'
# PATH_TO_TRAIN = '/data/speed/some-data/'
# PATH_TO_TRAIN = '/data/speed/all-data/'
PATH_TO_TRAIN = '/data/speed/training-data/'

PATH_TO_TEST = '/data/speed/test-data/'

# =============================================================================
# Data Reading
# =============================================================================

csvFilenames_train = [cwd + PATH_TO_TRAIN + f for f in os.listdir(cwd + PATH_TO_TRAIN)]
csvFilenames_test = [cwd + PATH_TO_TEST + f for f in os.listdir(cwd + PATH_TO_TEST)]

COLS = ["id", "vendor_id", "pu_month", "pu_dow", "pu_dom", "pu_h", "pc", "src", "dst", "pt", "distance", "temp",
        "area_src", "area_dst", "borough_src", "borough_dst", "crosses_bridge", "from_airport", "to_airport",
        "duration"]

# =============================================================================
# Codebook
# =============================================================================

# temp: temperature
# duration: integer in minutes
# src: source zone
# dst: destination zone
# distance: distance between centroids
# pt: payment type
# pc: passenger count

# =============================================================================
# Reading
# =============================================================================
df_train = pd.concat([pd.read_csv(f, names=COLS) for f in csvFilenames_train])

df_test = pd.concat([pd.read_csv(f, names=COLS) for f in csvFilenames_test])

df_train.insert(0, 'type', 'train')
df_test.insert(0, 'type', 'test')

df = pd.concat([df_train, df_test])

df.head(10)

# =============================================================================
# Data processing
# =============================================================================

# =============================================================================
# Data description
# =============================================================================

df.head(100)

df.columns

df.dtypes

df_train['distance'].head(10)

np.min(df_train['distance'])
np.max(df_train['distance'])
np.mean(df_train['distance'])

np.min(df_train['area_src'])

# =============================================================================
# Creation of variables
# =============================================================================

import math

# We assume that the zones are squares. For the cases of intrazonal trip, the distance is assumed to be the half of the side of a square
# df['travel_distance'] =

df.loc[df['distance'] == 0, 'travel_distance'] = np.sqrt(df.loc[df['distance'] == 0]['area_src']) / 2
df.loc[df['distance'] != 0, 'travel_distance'] = df.loc[df['distance'] != 0]['distance']

df['travel_distance'] = pd.to_numeric(df['travel_distance'])

np.mean(df['distance'])
np.mean(df['travel_distance'])

# Intra and interregional trips

b['travel_type'] = np.where(b['src'] == b['dst'], 'intra', 'inter')

# Regularize coefficients

Xs1 = pd.get_dummies(df_training['pu_h'], prefix='pu_h_dummy', drop_first=True)  # 23
Xs2 = pd.get_dummies(df_training['pu_dow'], prefix='pu_h_dow', drop_first=True)  # 6
Xs3 = pd.get_dummies(df_training['pu_month'], prefix='pu_h_month', drop_first=True)  # 6
Xs4 = pd.get_dummies(df_training['src'], prefix='src_dummy', drop_first=True)  # 6
Xs5 = pd.get_dummies(df_training['dst'], prefix='dst_dummy', drop_first=True)  # 6

XsC = df_training[['pt', 'pc']]

Xs = pd.concat([Xs1, Xs2, Xs3, Xs4, Xs5, XsC, df_training['log_travel_distance']], axis=1)

Xs.head

# =============================================================================
# Descriptive statistics
# =============================================================================
df['duration'].max()

# As expected, the distribution of travel times looks as a gamma

plt.hist(df['duration'])
plt.title('Travel times in NYC (< 2 hours)')
plt.xlabel('Travel time (minutes)')
plt.ylabel('Counts')
plt.show()

# Histogram by zone

# Mean travel time airport versus not

# =============================================================================
# Logarithmic model
# =============================================================================

df['log_duration'] = np.log(df['duration'])

df['log_travel_distance'] = np.log(df['travel_distance'])

df['log_speed'] = np.log((df['travel_distance']) / (df['duration']))

# Travel time by period of the day

# =============================================================================
# Training subset
# =============================================================================

import random

df_training = df[df['type'] == 'train']

df_training = df_training.iloc[0:1000000]

random_ids = random.sample(range(0, df_training.shape[0]), 10000)

random_ids

df_training = df_training.iloc[random_ids,]

df_training.head(10)

df_training.columns

df_training.shape

# =============================================================================
# Model estimation (INTERPRETABILITY)
# =============================================================================

import statsmodels.formula.api as smf

# Documentation: https://www.statsmodels.org/stable/index.html

results = smf.ols('duration ~ pt + pc', data=df_training).fit()
print(results.summary())

results1 = smf.ols('duration ~ pt + distance ', data=df_training).fit()
print(results1.summary())

results1b = smf.ols('duration ~ pt + distance + (temp < 0) ', data=df_training).fit()
print(results1b.summary())

# Checking if the travel tie in airports is higher in average

results2 = smf.ols('duration ~ pt + pc + to_airport', data=df_training).fit()

print(results2.summary())

# Model with interactions for zone

results3 = smf.ols('duration ~ pt + pc + to_airport + C(src) + C(dst)', data=df_training).fit()

print(results3.summary())

results4 = smf.ols('duration ~ pt + pc + to_airport + C(src) + C(dst) ', data=df_training).fit()

print(results4.summary())

results5 = smf.ols('duration ~ pt + pc + C(pu_month) + C(pu_dow) + C(pu_dom) + C(pu_h) + travel_distance + temp ',
                   data=df_training).fit()

print(results5.summary())

results6 = smf.ols(
    'duration ~ pt + pc + C(src) + C(dst) + travel_distance + temp + C(pu_month) + C(pu_dow) + C(pu_dom) + C(pu_h)'
    , data=df_training).fit()

print(results6.summary())

results7 = smf.ols('duration ~ pt + pc + travel_distance + temp + C(pu_month) + C(pu_dow) + C(pu_dom) + C(pu_h)'
                   , data=df_training).fit()

print(results7.summary())

results7b = smf.ols(
    'duration ~ pt + pc + travel_distance + temp + C(pu_month) + C(pu_dow) + C(pu_dom) + C(pu_h)+ C(src) + C(dst) '
    , data=df_training).fit()

print(results7b.summary())

results8 = smf.ols('duration ~ travel_distance+ C(src) + C(dst) '
                   , data=df_training).fit()

print(results8.summary())

### Final models

# COLS = ["id", "vendor_id", "pu_month", "pu_dow", "pu_dom", "pu_h", "pc", "src", "dst", "pt", "distance", "temp", "area_src", "area_dst", "borough_src", "borough_dst", "crosses_bridge", "from_airport", "to_airport", "duration"]

results_loglinear_traveltime = smf.ols(
    'log_duration ~ C(pu_month) + C(pu_dow) + C(pu_dom) + C(pu_h) + C(borough_src) + C(borough_dst) + C(crosses_bridge)  + C(from_airport) + C(to_airport) + C(pt) + pc  + (temp<0) + log_travel_distance'
    , data=df_training).fit()

print(results_loglinear_traveltime.summary())

results_linear = smf.ols(
    'duration ~  C(pu_month) + C(pu_dow) + C(pu_dom) + C(pu_h) + C(borough_src) + C(borough_dst) + C(crosses_bridge)  + C(from_airport) + C(to_airport) + C(pt) + pc  + (temp<0) + travel_distance'
    , data=df_training).fit()

print(results_linear.summary())

results_loglinear_speed = smf.ols(
    'log_speed ~ C(pu_month) + C(pu_dow) + C(pu_dom) + C(pu_h) + C(borough_src) + C(borough_dst) + C(crosses_bridge)  + C(from_airport) + C(to_airport) + C(pt) + pc  + (temp<0)'
    , data=df_training).fit()

print(results_loglinear_speed.summary())

# Time of the day variables

# =============================================================================
# REGULARIZATION: Lasso
# =============================================================================

from sklearn import preprocessing

# Number of zones

print(df_training.columns)

##There are 247 drop off zones at least when reading the first file
# df['dst'].unique().shape
#
##There are 201 pickup zones at least when reading the first file
# df['src'].unique().shape

# Xs = pd.get_dummies(df['dst'].unique())

# Xs1 = pd.get_dummies(df_training['dst'])
# Xs2 = pd.get_dummies(df_training['src'])

Xs1 = pd.get_dummies(df_training['pu_h'], drop_first=True)  # 23
Xs2 = pd.get_dummies(df_training['pu_dow'], drop_first=True)  # 6
Xs3 = pd.get_dummies(df_training['pu_month'], drop_first=True)  # 6
Xs4 = pd.get_dummies(df_training['src'], drop_first=True)  # 6
Xs5 = pd.get_dummies(df_training['dst'], drop_first=True)  # 6

XsC = df_training[['pt', 'pc']]

Xs = pd.concat([Xs1, Xs2, Xs3, Xs4, Xs5, XsC, df_training['log_travel_distance']], axis=1)

# Standarization of all variables
# Source: https://stats.stackexchange.com/questions/69568/whether-to-rescale-indicator-binary-dummy-predictors-for-lasso
# https://machinelearningmastery.com/rescaling-data-for-machine-learning-in-python-with-scikit-learn/

Xs = preprocessing.normalize(Xs)

Xs = Xs1

Xs.shape

# Xs = Xs.iloc[:,0:50]

Y = df_training['log_duration']
Y.shape
Xs.shape

# Lasso

# Sources:
# https://towardsdatascience.com/how-to-perform-lasso-and-ridge-regression-in-python-3b3b75541ad8
# https://scikit-learn.org/stable/auto_examples/exercises/plot_cv_diabetes.html

from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score


def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, Xs, Y, scoring="neg_mean_squared_error", cv=5))
    return (rmse)


# alphaList = [1e-20, 1e-18, 1e-16, 1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 0.05, 0.1, 0.3,1]

alphaList = [1e-20, 1e-18, 1e-16, 1e-15, 1e-10, 1e-8, 1e-4]

alphaList = [1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3]

# alphaList = [1e-5,1e-6,1e-4, 1e-3, 1e-2, 1e-1,1]

alphaList = [1e-6, 1e-5, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3]

alphaList = [1e-6, 1e-5, 1e-4, 2e-4, 3e-4, 4e-4, 1e-3]

# alphaList = [1e-20, 1e-18, 1e-16, 1e-11]

cv_lasso = [rmse_cv(LassoCV(alphas=[alpha], cv=2).fit(Xs, Y)).mean() for alpha in alphaList]

# cv_lasso.columns

# cv_lasso.sort_values(by=['Brand'], inplace=True)

cv_lasso = pd.Series(cv_lasso, index=alphaList)
cv_lasso.plot(title="Validation")
plt.xlabel("alpha")
plt.ylabel("rmse")

np.mean(Xs, axis=1)

alpha = 1
model_lasso = LassoCV(alphas=[1], cv=2).fit(Xs, Y)

rmse_cv(model_lasso)

lasso = Lasso()

# parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}

parameters = {'alpha': [1e-15, 1e-6, 1e-3, 1e-2, 2e-2, 0.05, 0.1, 0.3, 1, 2]}
# parameters = {'alpha': [1e-15, 1e-6, 1e-04, 1e-3]}

parameters = {'alpha': [1e-5, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1]}

# parameters = {'alpha': [1e-15]}
lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=2)

lasso_regressor.fit(Xs, Y)

print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

# Example

from sklearn import linear_model

clf = linear_model.Lasso(alpha=0.0003)

clf = linear_model.Lasso(alpha=0.001)

# clf = linear_model.Lasso(alpha=int(lasso_regressor.best_params_))

clf.fit(Xs, Y)

print(clf.coef_)

print(clf.intercept_)

clf.predict(Xs)

print("Mean squared error: %.2f"
      % mean_squared_error(Y, clf.predict(Xs)))

print(lasso_regressor.intercept_)

# Linear regression
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

regr = linear_model.LinearRegression()
regr.fit(Xs, Y)
print('Coefficients: \n', regr.coef_)

print("Mean squared error: %.2f"
      % mean_squared_error(Y, regr.predict(Xs)))

math.sqrt(mean_squared_error(Y, regr.predict(Xs)))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Y, regr.predict(Xs)))

# =============================================================================
# Model evaluation
# =============================================================================

import matplotlib.pyplot as plt
from math import sqrt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# In sample
pred_r3 = results3.predict(df)

fig, ax = plt.subplots()
ax.scatter(df['duration'], pred_r3)
ax.plot([df['duration'].min(), df['duration'].max()], [df['duration'].min(), df['duration'].max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

rms = sqrt(mean_squared_error(df['duration'], pred_r3))
rms

# Training and validation datasets

from matplotlib import pyplot as plt

X_train, X_validation, y_train, y_validation = train_test_split(df, df['duration'], test_size=0.2)

## RMSE
pred_r3_validation = results3.predict(X_validation)

fig, ax = plt.subplots()
ax.scatter(y_validation, pred_r3_validation)
ax.plot([y_validation.min(), y_validation.max()], [y_validation.min(), y_validation.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

rms_validation = sqrt(mean_squared_error(y_validation, pred_r3_validation))

rms_validation

np.mean(y_validation)

np.mean(pred_r3_validation)

np.mean(np.abs(y_validation - pred_r3_validation))

##Removing fixed effects

## RMSE
pred_r2_validation = results2.predict(X_validation)

rms_validation = sqrt(mean_squared_error(y_validation, pred_r2_validation))

rms_validation

rms_validation

## Fuera de muestra

a = df[df['type'] == 'test']

a.shape

b = a

b = a.iloc[1:5000000]

# pred_r5 = results_log1.predict(b)

# a.head
#
# a.head(10)
#
# b0 = a.iloc[0:3]
# b1 = a.iloc[1:10000001]
# b2 = a.iloc[10000001:a.shape[0]]
# b3 = a.iloc[a.shape[0]-1:a.shape[0]]
#
# b1.shape
# b2.shape
#
# pred_r5a0 = np.exp(results_loglinear_traveltime.predict(b0))
# pred_r5a = np.exp(results_loglinear_traveltime.predict(b1))
# pred_r5b = np.exp(results_loglinear_traveltime.predict(b2))
# pred_r5c = np.exp(results_loglinear_traveltime.predict(b3))
#
# b1.shape[0]+b2.shape[0]
# pred_r5a.shape[0]+pred_r5b.shape[0]
#
# pred_r5a.shape
# pred_r5b.shape
#
#
# pred_r5 = pd.concat([pred_r5a0,pred_r5a,pred_r5b,pred_r5c],axis = 0)
#
# pred_r5.shape
#
# pred_r5.shape
#
# pred_r5 = np.exp(results_loglinear_traveltime.predict(b))
#
# pred_r5.head


# Predicting travel_time
# pred_r5 = results7.predict(b)
pred_r5 = results_linear.predict(b)  # Only distance

##Predicting log speed
# pred_r5 = (b['travel_distance'])/np.exp(results_log1.predict(b))

# Predicting log travel_time
pred_r5 = np.exp(results_loglinear_traveltime.predict(b))

np.max(pred_r5)
np.min(pred_r5)

np.max(b['duration'])
np.min(b['duration'])

b['duration'].head

pred_r5.shape
b['duration'].shape

# mse = mean_squared_error(b['duration'].iloc[1:(b['duration'].shape[0]+1)], pred_r5.iloc[1:(pred_r5.shape[0]-1)])

mse = mean_squared_error(b['duration'], pred_r5)
print("Mean squared error: %.2f"
      % mse)

math.sqrt(mse)

# Interregional

b['travel_type'] = 0

b['travel_type'] = np.where(b['src'] == b['dst'], 'intra', 'inter')

pred_r5_intra = np.exp(results_log2.predict(b[b['travel_type'] == 'intra']))
pred_r5_inter = np.exp(results_log2.predict(b[b['travel_type'] == 'inter']))

math.sqrt(mean_squared_error(b[b['travel_type'] == 'intra']['duration'], pred_r5_intra))

math.sqrt(mean_squared_error(b[b['travel_type'] == 'inter']['duration'], pred_r5_inter))

# Predictions with regularized model

Xs1_testing = pd.get_dummies(b['pu_h'], drop_first=True)  # 23
Xs2_testing = pd.get_dummies(b['pu_dow'], drop_first=True)  # 6
Xs3_testing = pd.get_dummies(b['pu_month'], drop_first=True)  # 6
Xs4_testing = pd.get_dummies(b['src'], drop_first=True)  # 6
Xs5_testing = pd.get_dummies(b['dst'], drop_first=True)  # 6

XsC_testing = b[['pt', 'pc']]

Xs_testing = pd.concat(
    [Xs1_testing, Xs2_testing, Xs3_testing, Xs4_testing, Xs5_testing, XsC_testing, b['log_travel_distance']], axis=1)

Xs_testing.shape

Xs.shape

preds = pd.DataFrame({"preds": np.exp(clf.predict(Xs_testing)), "true": Xs_testing['duration']})

np.exp(clf.predict(Xs_testing)).head

Xs_testing

# =============================================================================
# Plots
# =============================================================================

#### a) Relationship between travel time and distance (slope is the reciprocal of the speed)


#### b) Predicted travel time versus observed travel times


#### c) Regularization (Lambda versus RMSE)

#### d) Lasso coefficient as level of importance (Lambda versus RMSE)


# =============================================================================
# Tables
# =============================================================================

### Linear regression estimation results

## There will be 6 columns:

# C1)






