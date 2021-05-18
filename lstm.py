#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 01:45:06 2021

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
2) Data Processing

*********************************************************************
"""
from sklearn.preprocessing import MinMaxScaler

def process_data(dataset, obs):
    # Importing training set
    df_train = pd.read_csv(dataset)
    d = len(df_train)
    training_set = df_train.iloc[:, 1:2]

    # Feature Scaling
    sc = MinMaxScaler(feature_range = (0,1))  
    training_set_scaled = sc.fit_transform(training_set) #Normalization Min = 0 and Max = 1

    print(training_set_scaled)
    
    X_train = []
    y_train = []
    for i in range(obs, d): #quantos dias de cada vez ? obs ,  tamanho do dataset 1258
        X_train.append(training_set_scaled[i-obs:i, 0]) # Pegar nos ultimos obs (do 0 ao obs-1) para prever o obs+1(obs no indice)
        y_train.append(training_set_scaled[i, 0]) #Posição obs
    X_train, y_train = np.array(X_train),  np.array(y_train)
    
    # Reshaping
    X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1], 1)) #3D
    
    return X_train, y_train

X_train, y_train = process_data('training_set.csv', 60)
print(X_train)
print(y_train)


"""
*********************************************************************
3) LSTM MODEL

*********************************************************************
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


def model_lstm(X_train, middle_layers, neurons, amnesia, optm, loss_funct):
    # Initialising the RNN
    regressor = Sequential()
    # Adding the first LSTM layer and some dropout regularization
    regressor.add(LSTM(units = neurons, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(amnesia))
    i = 0
    while i < middle_layers:
        regressor.add(LSTM(units = neurons, return_sequences = True))
        regressor.add(Dropout(amnesia))
        i += 1
    
    regressor.add(LSTM(units = neurons, return_sequences = False))
    regressor.add(Dropout(amnesia))
    
    #Adding the output layer 
    regressor.add(Dense(units = 1))
    # Compiling the RNN 
    regressor.compile(optimizer = optm, loss = loss_funct)
    
    print("\n********************************************")
    print("LSTM")
    print("********************************************")
    print(regressor.summary())
    return regressor

regressor = model_lstm(X_train,2,50,0.2,"adam",'mean_squared_error')
    
def fit_model_lstm(regressor, X_train, y_train, epochs, batch_size):
    regressor.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)
    return regressor

regressor = fit_model_lstm(regressor, X_train, y_train, 2, 1000)
    

"""
*********************************************************************
4) Vizualization Results

*********************************************************************
"""


