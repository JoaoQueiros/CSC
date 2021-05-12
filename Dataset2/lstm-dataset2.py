import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

#for replicability purposes
tf.random.set_seed(91195003)
np.random.seed(91190530)
#for an easy reset backend session state
tf.keras.backend.clear_session()

#Load dataset
def load_dataset(path=r'GlobalTemperatures2.csv'):
    return pd.read_csv(path)

#split data training and validation sets
def split_data(training, perc=10):
    train_idx = np.arange(0,int(len(training)*(100-perc)/100))
    val_idx = np.arange(int(len(training)*(100-perc)/100+1), len(training))
    return train_idx, val_idx

#Preaparing the data for the LSTM
def prepare_data(df):
    df_aux = df.drop(columns=['dt','LandAverageTemperatureUncertainty', 'LandMaxTemperature', 'LandMaxTemperatureUncertainty', 'LandMinTemperature','LandMinTemperatureUncertainty','LandAndOceanAverageTemperature','LandAndOceanAverageTemperatureUncertainty'], inplace=False)
    #number of confirmed cases per day
    df_aux.dropna(inplace=True)
    return df_aux

def data_normalization(df, norm_range=(-1,1)):
    #[-1, 1] for LSTM due to the internal use of tanh by the memory cell
    scaler = MinMaxScaler(feature_range=norm_range)
    df[['LandAverageTemperature']] = scaler.fit_transform(df[['LandAverageTemperature']])
    return scaler

#Plot time series data
def plot_confirmed_cases(data):
    plt.figure(figsize=(8,6))
    plt.plot(range(len(data)), data)
    plt.title('Confirmed Cases of COVID-19')
    plt.ylabel('Cases')
    plt.xlabel('Days')
    plt.show()

#Preparing the dataset for the LSTM
def to_supervised(df, timesteps):
    data = df.values
    X, y = list(), list()
    #iterate over the training set to create X and y
    dataset_size = len(data)
    for curr_pos in range(dataset_size):
        #end of the input sequence is the current position + the number of timesteps of the input sequence
        input_index = curr_pos + timesteps
        #end of the labels corresponds to the end of the input sequence + 1
        label_index = input_index + 1
        #if we have enough data for this sequence
        if label_index < dataset_size:
            X.append(data[curr_pos:input_index, :])
            y.append(data[input_index:label_index, 0])
    #using np.float32 for GPU performance
    return np.array(X).astype('float32'), np.array(y).astype('float32')

#Building the model
def rmse(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)))

def build_model(timesteps, features, h_neurons=64, activation='tanh'):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.LSTM(h_neurons, input_shape=(timesteps, features)))
    model.add(tf.keras.layers.Dense(h_neurons, activation=activation))
    model.add(tf.keras.layers.Dense(1, activation='relu'))

    #model summary (and save it as PNG)
    tf.keras.utils.plot_model(model, 'covid19_model.png', show_shapes=True)
    return model

#Vizualizing Learning Curves
def plot_learning_curves(history, epochs, approach):
    #accuracies and losses
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)
    #creating figure
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training/Validation Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training/Validation Loss ', approach)
    plt.show()
    #plot learning curves
    plot_learning_curves(history, epochs)

#Compiling and fit the model
def compile_and_fit(model, epochs, batch_size):
    #compile
    model.compile(loss = rmse, optimizer = tf.keras.optimizers.Adam(), metrics = ['mae', rmse])
    #fit
    hist_list = list()
    loss_list = list()
    #Time Series Cross Validator
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    for train_index, test_index in tscv.split(X):
        train_idx, val_idx = split_data(train_index, perc=10) #further split into training and validation sets
        #build data
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_index], y[test_index]
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=epochs, batch_size=batch_size, shuffle=False)
        metrics = model.evaluate(X_test, y_test)
        hist_list.append(history)
        loss_list.append(metrics[2])
    #plot_learning_curves(hist_list, epochs, approach='history')
    #plot_learning_curves(loss_list, epochs, approach='loss')
    return model, hist_list, loss_list

#Recursive Multi-Step Forecast!!!
def forecast(model, df, timesteps, multisteps, scaler):
    input_seq = df[-timesteps:].values #getting the last sequence of known value
    inp = input_seq
    forecasts = []
    #multisteps tells us how many iterations we want to perform, i.e., how many days we want to predict
    for step in range(multisteps):
        inp = inp.reshape(1, timesteps, 1) # (1 sequence, n timesteps, 1 variable)
        #the next six lines are for you to implement :)
        pred = model.predict(inp)
        pred_inverse_scale = scaler.inverse_transform(pred)
        forecasts.append(pred_inverse_scale[0][0])
        inp = np.append(inp[0], pred)
        inp = inp[-timesteps:]
    return forecasts


def plot_forecast(data, forecasts):
    newdata = data.iloc[3000:len(data)]
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(newdata)), newdata, color='green', label='Confirmed')
    plt.plot(range(len(newdata) - 1, len(newdata) + len(forecasts) - 1), forecasts, color='red', label='Forecasts')
    plt.title('Temperatura')
    plt.ylabel('Graus')
    plt.xlabel('Meses')
    plt.legend()
    plt.show()

#Main Execution
timesteps = 12 #number of days that make up a sequence
univariate = 1 #number of features used by the model (using conf. cases to predict conf. cases)
multisteps = 24 #number of days to forecast – we will forecast the next 5 days
cv_splits = 3 #time series cross validator
epochs = 50
batch_size = 12 #7 sequences of 5 days - which corresponds to a window of 7 days in a batch

#the dataframes
df_raw = load_dataset()
df_data = prepare_data(df_raw)
print(df_data)
df = df_data.copy()
plot_confirmed_cases(df_data) #the plot you saw previously
scaler = data_normalization(df) #scaling data to [-1, 1]
#our supervised problem
X, y = to_supervised(df, timesteps)
print("Training shape:", X.shape)
print("Training labels shape:", y.shape)
#fitting the model
model = build_model(timesteps, univariate)
model, hist_list, loss_list = compile_and_fit(model, epochs, batch_size)
#...
#Now that we have “tuned” our model, we should retrain it with all the available data
#(as we did in SBS) and obtain real predictions for tomorrow, the day after, …
#We want to forecast the next five days after today!
#We just need to:
model = build_model(timesteps, univariate)
model.compile(loss = rmse, optimizer = tf.keras.optimizers.Adam(), metrics = ['mae', rmse])
model.fit(X, y, epochs=epochs, batch_size=batch_size, shuffle=False)

#Recursive Multi-Step Forecast!!!
forecasts = forecast(model, df, timesteps, multisteps, scaler)
plot_forecast(df_data, forecasts)