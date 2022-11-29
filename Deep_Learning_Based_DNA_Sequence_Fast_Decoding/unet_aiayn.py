from __future__ import print_function

import os
import sys
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv1D, MaxPooling1D, Conv2DTranspose, Lambda, BatchNormalization, LSTM, MultiHeadAttention, Dropout, LayerNormalization
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.layers import ZeroPadding1D
from keras import backend as K

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

ss=10

def crossentropy_cut(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f= tf.clip_by_value(y_pred_f, 1e-7, (1. - 1e-7))
    mask=K.greater_equal(y_true_f,-0.5)
    losses = -(y_true_f * K.log(y_pred_f) + (1.0 - y_true_f) * K.log(1.0 - y_pred_f))
    losses = tf.boolean_mask(losses, mask)
    masked_loss = tf.reduce_mean(losses)
    return masked_loss

def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same'):
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    mask=K.cast(K.greater_equal(y_true_f,-0.5),dtype='float32')
    intersection = K.sum(y_true_f * y_pred_f * mask)
    return (2. * intersection + ss) / (K.sum(y_true_f * mask) + K.sum(y_pred_f * mask) + ss)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def pcc(layer_in, num_filter, size_kernel, activation='relu', padding='same'):
    x = MaxPooling1D(pool_size=2)(layer_in)
    x = BatchNormalization()(Conv1D(num_filter,size_kernel,activation=activation,padding=padding)(x))
    x = BatchNormalization()(Conv1D(num_filter,size_kernel,activation=activation,padding=padding)(x))
    return x

def ucc(layer_in1,layer_in2, num_filter, size_kernel, activation='relu', padding='same'):
    x = concatenate([Conv1DTranspose(layer_in1,num_filter,2,strides=2,padding=padding), layer_in2], axis=2)
    x = BatchNormalization()(Conv1D(num_filter,size_kernel,activation=activation,padding=padding)(x))
    x = BatchNormalization()(Conv1D(num_filter,size_kernel,activation=activation,padding=padding)(x))
    return x

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    return x + res

def get_unet(the_lr=1e-3, num_class=1, num_channel=5, size=10240):
    inputs = Input((size, num_channel))

    # unet parameters
    num_blocks=5 # 2^20 -> 2^8=256
    initial_filter=15
    scale_filter=1.5
    size_kernel=7
    activation='relu'
    padding='same'

    # transformer parameters
    num_transformer_blocks=4
    head_size=256
    num_heads=4
    ff_dim=4
    dropout=0.

    layer_down=[]
    layer_up=[]

    conv0 = BatchNormalization()(Conv1D(initial_filter, size_kernel, \
        activation=activation, padding=padding)(inputs))
    conv0 = BatchNormalization()(Conv1D(initial_filter, size_kernel, \
        activation=activation, padding=padding)(conv0))

    layer_down.append(conv0)
    num=initial_filter

    # unet compress
    for i in range(num_blocks):
        num=int(num * scale_filter)
        the_layer=pcc(layer_down[i], num, size_kernel, activation=activation, padding=padding)
        layer_down.append(the_layer)

    # insert transformer here
    for _ in range(num_transformer_blocks):
        the_layer = transformer_encoder(the_layer, head_size, num_heads, ff_dim, dropout)

    layer_up.append(the_layer)
    # unet decompress
    for i in range(num_blocks):
        num=int(num / scale_filter)
        the_layer=ucc(layer_up[i],layer_down[-(i+2)],num, size_kernel, activation=activation, padding=padding)
        layer_up.append(the_layer)

    convn = Conv1D(num_class, 1, activation='sigmoid', padding=padding)(layer_up[-1])

    model = Model(inputs=[inputs], outputs=[convn])
    return model


