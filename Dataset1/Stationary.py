#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 00:07:40 2021

@author: luis
"""

"""
*********************************************************************
1) Import Packages

*********************************************************************
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

"""
*********************************************************************
2) Import Training Set

*********************************************************************
"""
df_train = pd.read_csv('training_set.csv')
training_set = df_train.iloc[:, 0:2]

training_set.plot()
training_set.hist()

X = training_set["n_incidents"].values
split = round(len(X) / 2)
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))


"""
*********************************************************************
3) Check if Stationary with ADF Statistic (Augmented Dickey-Fuller test) 

Null Hypothesis (H0): If failed to be rejected, it suggests the time series has a unit root, meaning it is non-stationary. It has some time dependent structure.
Alternate Hypothesis (H1): The null hypothesis is rejected; it suggests the time series does not have a unit root, meaning it is stationary. It does not have time-dependent structure.

p-value > 0.05: Fail to reject the null hypothesis (H0), the data has a unit root and is non-stationary.
p-value <= 0.05: Reject the null hypothesis (H0), the data does not have a unit root and is stationary.

*********************************************************************
"""
print("\n********************************************")
print("Adf Statistics")
print("********************************************")
from statsmodels.tsa.stattools import adfuller
X = training_set["n_incidents"].values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

"""
*********************************************************************
4) Logaritmization

*********************************************************************
"""
ts = training_set
ts['n_incidents'] = np.log2(ts['n_incidents'])
ts.plot()
training_set.hist()
print("\n********************************************")
print("Adf Statistics")
print("********************************************")
from statsmodels.tsa.stattools import adfuller
X = ts["n_incidents"].values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))




