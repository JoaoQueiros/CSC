import logging
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

import pandas as pd
logging.getLogger('tensorflow').setLevel(logging.ERROR)
#for replicability purposes set seed
tf.random.set_seed(1234)
#for an easy reset backend session state
tf.keras.backend.clear_session()


#hyperparameters
epochs = 6
batch_size = 32
learning_rate = 1e-3
output_neurons = 10




# =============================================================================
#
#         PerceptronLayer Class
#
# =============================================================================
class PerceptronLayer(Layer):
    '''
    The constructor in an object oriented perspective.
    Called when an object is created, allowing the class to initialize the attributes of a class.
    neurons corresponds to the number of neurons in this perceptron layer
    '''

    def __init__(self, neurons=16, **kwargs):
        super(PerceptronLayer, self).__init__(**kwargs)
        self.neurons = neurons

    '''
    We use the build function to deferr weight creation until the shape of the inputs is known
    '''

    def build(self, input_shape):
        # ---
        # weights_init = tf.random_normal_initializer()
        # self.w = tf.Variable(initial_value=weights_init(shape=(????, ????), dtype='float32'), trainable=True)
        # bias_init = tf.zeros_initializer()
        # self.b = tf.Variable(initial_value=bias_init(shape=(????,), dtype='float32'), trainable=True)
        # you will need to assert a fact, i.e., that weights and bias are to be automatically tracked by our layer
        # for example: assert perceptron.weights == [perceptron.w, perceptron.b]
        # ---
        # however, we can use a shortcut for adding weights to a layer
        # TODO: set the correct shape for the weights and the bias
        # weights = nº inputs * nº neuronios
        self.w = self.add_weight(name='w', shape=(input_shape[-1], self.neurons), initializer='random_normal', trainable=True)
        # nº Bias = nº de neuronios
        self.b = self.add_weight(name='b', shape=(self.neurons,), initializer='random_normal', trainable=True)

    '''
    Implements the function call operator (when an instance is used as a function).
    It will automatically run build the first time it is called, i.e., layer's weights are created dynamically
    '''
    #resultado do Call, tem de ser um arrya de 1 * (nº de neuronios), em cada posicao desse arry
    #tem o resultado da conta Z = Inputs*W(pesos) + B(bias)
    #w tem este formato = [[1,2,3 (se 3 pesos entrar no neuronio 1],...,[]]
    #b tem este formato = [1(bias no neuronio 1),...,(bias no ultimo neuronio)]
    #Se 3 inputs e 4 neuronios, I*W = [1*4] ao qual lhe somo b que é um array [1*4]
    def call(self, inputs):
        # TODO: return the perceptron result
        return tf.matmul(inputs, self.w) + self.b

    '''
    Enable serialization on our perceptron layer
    '''

    def get_config(self):
        config = super(PerceptronLayer, self).get_config()
        config.update({'neurons': self.neurons})
        return config





# =============================================================================
#
#         Multilayer Perceptron Class
#
# =============================================================================
class MultilayerPerceptron(Model):
    '''
    The Layers of our MLP (with a fixed number of neurons)
    '''

    def __init__(self, output_neurons=10, name='multilayerPerceptron', **kwargs):
        super(MultilayerPerceptron, self).__init__(name=name, **kwargs)
        self.perceptron_layer_1 = PerceptronLayer(16)
        self.perceptron_layer_2 = PerceptronLayer(32)
        self.perceptron_layer_3 = PerceptronLayer(output_neurons)

    '''
    Layers are recursively composable, i.e., 
    if you assign a Layer instance as attribute of another Layer, the outer layer will start tracking the weights of the inner layer.
    Remember that the build of each layer is called automatically (thus creating the weights).
    '''

    def feed_model(self, input_data):
        x = self.perceptron_layer_1(input_data)
        # activation function applied to the output of the perceptron layer
        x = tf.nn.relu(x)
        # the output, now normalized, is fed as input to the second perceptron layer
        x = self.perceptron_layer_2(x)
        # again, activation function applied to the output of the second perceptron layer
        x = tf.nn.relu(x)
        # which, again, is fed as input to the third layer, which returns its output
        # TODO: logits should be what?
        logits = self.perceptron_layer_3(x)
        # the output of the last layer going over a softmax activation
        # so, we will not be outputting logits but "probabilities"
        return self.softmax(logits)  # equivalent of tf.nn.softmax(logits)

    """
    Compute softmax values for the logits
    """

    def softmax(self, logits):
        """Compute softmax values for each sets of scores in x."""
        return tf.math.exp(logits) / tf.math.reduce_sum(tf.math.exp(logits), axis=1, keepdims=True)

    def print_trainable_weights(self):
        print('Weights:', len(self.weights))
        print('Trainable weights:', len(self.trainable_weights))
        print('Non-trainable weights:', len(self.non_trainable_weights))

    def call(self, input_data):
        # TODO: here, we want to feed the model and receive its output (i.e, the output of the last layer)
        probs = self.feed_model(input_data)
        return probs








# =============================================================================
#
#         Main Execution
#
# =============================================================================

'''
Importing data
'''
def import_data():
    # load mnist training and test data
    raw_data = pd.read_csv(r'GlobalTemperatures2.csv')
    
    y = raw_data.pop('LandAverageTemperature')
    x = raw_data.pop('dt')
    print(y)
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=42)


    # reshape the input to have a list of self.batch_size by 28*28 = 784; and normalize (/255)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2]).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2]).astype('float32') / 255
    # reserve 5000 samples for validation
    x_validation = x_train[-5000:]
    y_validation = y_train[-5000:]
    # do not use those same 5000 samples for training!
    x_train = x_train[:-5000]
    y_train = y_train[:-5000]

    # create dataset iterator for training
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    # shuffle in intervals of 1024 and slice in batchs of batch_size
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    # create the validation dataset
    validation_dataset = tf.data.Dataset.from_tensor_slices((x_validation, y_validation))
    validation_dataset = validation_dataset.batch(batch_size)
    return train_dataset, validation_dataset, x_test, y_test



'''
Preparing the model, the optimizers, the loss function and some metrics
'''
def prepare_model():
    mlp = MultilayerPerceptron(output_neurons=output_neurons)
    #instantiate an optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    #instantiate a loss object (from_logits=False as we are applying a softmax activation over the last layer)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    #TODO: using a metric too
    train_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    val_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    return mlp, optimizer, loss_object, train_metric, val_metric






'''
Define a low level fit and predict making use of the tape.gradient
'''
def low_level_fit_and_predict():
    # manually, let's iterate over the epochs and fit ourselves
    for epoch in range(epochs):
        print('Epoch %d/%d' % (epoch + 1, epochs))

        # to store loss values
        loss_history = []

        # iterate over all batchs
        for step, (x_batch, y_batch) in enumerate(train_dataset):
            # use a gradien tape to save computations to calculate gradient later
            with tf.GradientTape() as tape:
                # running the forward pass of all layers
                # operations being recorded into the tape
                # TODO: what is the input of the model?
                probs = mlp(x_batch)
                # computing the loss for this batch
                # how far are we from the correct labels?
                # TODO: what is the input of the losso object?
                loss_value = loss_object(y_batch, probs)

            # store loss value
            loss_history.append(loss_value.numpy().mean())
            # use the tape to automatically retrieve the gradients of the trainable variables
            # with respect to the loss
            gradients = tape.gradient(loss_value, mlp.trainable_weights)
            # running one step of gradient descent by updating (going backwards now)
            # the value of the trainable variables to minimize the loss
            optimizer.apply_gradients(zip(gradients, mlp.trainable_weights))
            # Update training metric.
            train_metric(y_batch, probs)

            # log every n batches
            if step % 200 == 0:
                print('Step %s. Loss Value = %s; Mean loss = %s' % (step, str(loss_value.numpy()), np.mean(loss_history)))

        # display metrics at the end of each epoch
        train_accuracy = train_metric.result()
        print('Training accuracy for epoch %d: %s' % (epoch + 1, float(train_accuracy)))
        # reset training metrics (at the end of each epoch)
        train_metric.reset_states()

        # run a validation loop at the end of each epoch
        for x_batch_val, y_batch_val in validation_dataset:
            val_probs = mlp(x_batch_val)
            # update val metrics
            val_metric(y_batch_val, val_probs)

        val_acc = val_metric.result()
        val_metric.reset_states()
        print('Validation accuracy for epoch %d: %s' % (epoch + 1, float(val_acc)))

    # now predict
    print('\nGenerating predictions for ten samples...')
    predictions = mlp(x_test[:10])
    print('Predictions shape:', predictions.shape)

    for i, prediction in enumerate(predictions):
        # tf.argmax returns the INDEX with the largest value across axes of a tensor
        predicted_value = tf.argmax(prediction)
        label = y_test[i]
        print('Predicted a %d. Real value is %d.' % (predicted_value, label))




'''
Define a high level fit and predict making use tf.Keras APIs
'''
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='my_model_{epoch}_{val_loss:.3f}.h5',#path where to save model
        save_best_only=True,#overwrite the current checkpoint if and only if
        monitor='val_loss',#the val_loss score has improved
        save_weights_only=False,#if True, only the weigths are saved
        verbose=1,#verbosity mode
        period=5#save ony at the fifth epoch (5 em 5 epocas)
    )
]

def high_level_fit_and_predict():
    # shortcut to compile and fit a model!
    # able to do this because our model subclasses tf.keras.Model
    mlp.compile(optimizer, loss=loss_object, metrics=[train_metric])
    # since the train_dataset already takes care of batching, we don't pass a batch_size argument
    # passing validation data for monitoring validation loss and metrics at the end of each epoch
    # TODO: what is the input of fit?
    history = mlp.fit(train_dataset, validation_data=validation_dataset, epochs=epochs, callbacks=callbacks)
    # print('\nHistory values per epoch:', history.history)

    # evaluating the model on the test data
    print('\nEvaluating model on test data...')
    scores = mlp.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    print('Evaluation %s: %s' % (mlp.metrics_names, str(scores)))

    # finally, generating predictions (the output of the last layer)
    print('\nGenerating predictions for ten samples...')
    # TODO: what to predict?
    predictions = mlp.predict(x_test[:10])
    # now, for each prediction in predictions, get the value with higher "probability"
    # look at the shape, it is as (3, 10). For each prediction, we have the prob of beeing 0, beeing 1, etc...
    # we now choose the index of the list with higher "probability"
    # if pos=3 is the one with higher probability it means it predicts a 3
    print('Predictions shape:', predictions.shape)
    for i, prediction in enumerate(predictions):
        # tf.argmax returns the INDEX with the largest value across axes of a tensor
        predicted_value = tf.argmax(prediction)
        label = y_test[i]
        print('Predicted a %d. Real value is %d.' % (predicted_value, label))


# Press the green button in the gutter to run the script.
    # load data
train_dataset, validation_dataset, x_test, y_test = import_data()
    # init our model
mlp, optimizer, loss_object, train_metric, val_metric = prepare_model()
    # use low-level or high-level fit and predict
#low_level_fit_and_predict()
high_level_fit_and_predict()

