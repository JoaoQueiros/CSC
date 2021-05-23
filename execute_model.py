#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 03:04:47 2021

@author: luis
"""

import tensorflow as tf
import lstm as lstm_model
dataset = "training_set.csv"
test_data = "test_set.csv"

'''
***********CHOOSE FILE*********
'''

path = "lstm_model"
obs = 80 # Number of last observations
middle_layers = 2 #Number of layers
neurons = 20 # Number of neurons for layers
f_activation = "tanh" # Activation Function 
amnesia = 0.2 # drop out 
optm = "adam" # optimizer
#loss_funct = "mean_squared_error" # loss function
epochs = 30 # epochs
batch_size = 1000 # batch size

file = str(path)+"/obs_"+str(obs)+"_ml_"+str(middle_layers)+"_n_"+str(neurons)+"_f_"+str(f_activation)+"_am_"+str(amnesia)+"_opt_"+str(optm)+"_ep_"+str(epochs)+"_bs_"+str(batch_size)


'''
***********IMPORT MODEL*********
'''

model = tf.keras.models.load_model(file)
print(model.summary())

'''
***********PREDICT*********
'''
X_train, y_train, df_train, obs, sc = lstm_model.process_data(dataset, obs)
# Predict test
predicted, real_Data = lstm_model.results_Lstm(test_data, df_train, obs, sc, model = file)
# Vizualize Results 
lstm_model.vizualize(predicted, real_Data)
