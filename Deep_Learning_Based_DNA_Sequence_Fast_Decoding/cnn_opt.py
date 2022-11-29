import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def build_model(hp):
    # create model object
    model = keras.Sequential([
    #adding first convolutional layer
    layers.Conv1D(
        #adding filter
        filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
        # adding filter size or kernel size
        kernel_size=hp.Choice('conv_1_kernel', values = [2,20]),
        #activation function
        activation='relu',
        padding='same',
        input_shape=(10240, 5)),
    # adding second convolutional layer
    layers.Conv1D(
        #adding filter
        filters=hp.Int('conv_2_filter', min_value=32, max_value=128, step=16),
        #adding filter size or kernel size
        kernel_size=hp.Choice('conv_2_kernel', values = [2,20]),
        #activation function
        activation='relu',
        padding='same'),
    layers.Conv1D(
        #adding filter
        filters=hp.Int('conv_3_filter', min_value=32, max_value=128, step=16),
        #adding filter size or kernel size
        kernel_size=5,
        #activation function
        activation='relu',
        padding='same'),
    # output layer
    layers.Dense(1, activation='sigmoid')
    ])
    return model

#importing random search
from kerastuner import RandomSearch
#creating randomsearch object
tuner = RandomSearch(build_model,
                    objective='val_accuracy',
                    max_trials = 5)
# search best parameter
tuner.search(train_df,train_labl,epochs=3,validation_data=(train_df,train_labl))
