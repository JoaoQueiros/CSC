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
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import os
"""
*********************************************************************
2) Data Processing

*********************************************************************
"""

def process_data(dataset, obs):
    #Select number of last observations
    obs = obs
    # Importing training set
    df_train = pd.read_csv(dataset)
    d = len(df_train)
    training_set = df_train.iloc[:, 1:2]

    # Feature Scaling
    sc = MinMaxScaler(feature_range = (0,1))  
    training_set_scaled = sc.fit_transform(training_set) #Normalization Min = 0 and Max = 1

    
    X_train = []
    y_train = []
    for i in range(obs, d): #quantos dias de cada vez ? obs ,  tamanho do dataset
        X_train.append(training_set_scaled[i-obs:i, 0]) # Pegar nos ultimos obs (do 0 ao obs-1) para prever o obs+1(obs no indice)
        y_train.append(training_set_scaled[i, 0]) #Posição obs
    X_train, y_train = np.array(X_train),  np.array(y_train)
    
    # Reshaping
    X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1], 1)) #3D
    
    return X_train, y_train, df_train, obs, sc


"""
*********************************************************************
3) LSTM MODEL

*********************************************************************
"""


def model_lstm(X_train, middle_layers, neurons, f_activation, amnesia, optm, loss_funct):
    # Initialising the RNN
    regressor = Sequential()
    # Adding the first LSTM layer and some dropout regularization
    regressor.add(LSTM(units = neurons, activation=f_activation, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(amnesia))
    i = 0
    
    #Number of middle_layers
    while i < middle_layers:
        regressor.add(LSTM(units = neurons, activation=f_activation, return_sequences = True))
        regressor.add(Dropout(amnesia))
        i += 1
    
    regressor.add(LSTM(units = neurons, activation=f_activation, return_sequences = False))
    regressor.add(Dropout(amnesia))
    
    #Adding the output layer 
    regressor.add(Dense(units = 1, activation='relu'))
    # Compiling the RNN 
    regressor.compile(optimizer = optm, loss = loss_funct)
    
    print("\n********************************************")
    print("LSTM")
    print("********************************************")
    print(regressor.summary())
    #plot_model(regressor, to_file='lstm.png', show_shapes=True, show_layer_names=True)
    return regressor

    
def fit_model_lstm(regressor, X_train, y_train, checkpoint_path, save_file, epochs, batch_size):
    checkpoint_path = checkpoint_path
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        save_freq=5*batch_size)
    
    # Save the weights using the `checkpoint_path` format
    regressor.save_weights(checkpoint_path.format(epoch=0))
    
    history = regressor.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, callbacks=[cp_callback])

    regressor.save(save_file)
    
    
"""
*********************************************************************
4)  Results

*********************************************************************
"""

def results_Lstm(test_data, dataset_train, obs, sc, model):
    #import model
    model = tf.keras.models.load_model(model)
    #Getting Real days
    dataset_test = pd.read_csv(test_data)
    real_Data = dataset_test.iloc[:, 1:2]
    #print(real_Data)
    
    #Getting the predict
    dataset_total = pd.concat((dataset_train['n_incidents'], dataset_test['n_incidents']), axis = 0)
    
    inputs = dataset_total[len(dataset_total)-len(dataset_test) - obs:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(obs, obs+len(real_Data)): 
        X_test.append(inputs[i-obs:i, 0]) 
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1], 1)) 
    
    predicted = model.predict(X_test)
    predicted = sc.inverse_transform(predicted)
    
    return predicted, real_Data
    

"""
*********************************************************************
5)  Vizualization Results

*********************************************************************
"""
def vizualize(predicted, real_Data):
    plt.plot(real_Data, color = 'red', label = 'Real values')
    plt.plot(predicted, color = 'green', label = 'predicted')
    plt.title('Number of Incidents')
    plt.xlabel('Time')
    plt.ylabel('Number of incidents')
    plt.legend()
    plt.show()



"""
*********************************************************************
6)  LSTM MODEL FUNCTION TRAINING

*********************************************************************
"""
dataset = "training_set.csv"
test_data = "test_set.csv"
checkpoint_path = "Training/cp-{epoch:04d}.ckpt"
#save_file = "lstm_model"
""" 
*********************Parameters******************
"""
obs = 80 # Number of last observations
middle_layers = 2 #Number of layers
neurons = 20 # Number of neurons for layers
f_activation = "tanh" # Activation Function 
amnesia = 0.2 # drop out 
optm = "adam" # optimizer
loss_funct = "mean_squared_error" # loss function
epochs = 10 # epochs
batch_size = 1000 # batch size

save_file = "lstm_model/obs_"+str(obs)+"_ml_"+str(middle_layers)+"_n_"+str(neurons)+"_f_"+str(f_activation)+"_am_"+str(amnesia)+"_opt_"+str(optm)+"_ep_"+str(epochs)+"_bs_"+str(batch_size)

def lstm_model_training_function(dataset, test_data, checkpoint_path, save_file, obs, middle_layers, neurons, f_activation, amnesia, optm, loss_funct, epochs, batch_size):
    #process data
    X_train, y_train, df_train, obs, sc = process_data(dataset, obs)
    #model lstm
    regressor = model_lstm(X_train, middle_layers, neurons, f_activation, amnesia, optm, loss_funct) 
    # fit model and Save
    fit_model_lstm(regressor, X_train, y_train, checkpoint_path, save_file, epochs, batch_size)
    # Predict test
    predicted, real_Data = results_Lstm(test_data, df_train, obs, sc, model = save_file)
    # Vizualize Results 
    vizualize(predicted, real_Data)
    
#train: 
    
#lstm_model_training_function(dataset, test_data, checkpoint_path, save_file, obs, middle_layers, neurons, f_activation, amnesia, optm, loss_funct, epochs, batch_size)
    


