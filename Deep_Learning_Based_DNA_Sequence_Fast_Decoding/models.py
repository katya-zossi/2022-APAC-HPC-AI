import os
import tensorflow as tf
#import unet
import unet_aiayn
#import attention_is_all_you_need
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def cnn_model(max_len, vocab_size):
    model = keras.Sequential([
        layers.InputLayer(input_shape=(max_len, vocab_size)),
        layers.Conv1D(32, 17, padding='same', activation='relu'),
        layers.Conv1D(64, 11, padding='same', activation='relu'),
        layers.Conv1D(128, 5, padding='same', activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    return model


## Step 1: Implement your own model below
"""
def unet_model(max_len, vocab_size):
    model = unet.get_unet(the_lr=1e-3, num_class=1, num_channel=vocab_size, size=max_len)
    return model
"""

def transformer_model(max_len, vocab_size):
    model = unet_aiayn.get_unet(the_lr=1e-3, num_class=1, num_channel=vocab_size, size=max_len)
    return model



## Step 2: Add your model name and model initialisation in the model dictionary below

def return_model(model_name, max_len, vocab_size):
    model_dic={'cnn': cnn_model(max_len, vocab_size),
               'aiayn': transformer_model(max_len, vocab_size)}
    return model_dic[model_name]



