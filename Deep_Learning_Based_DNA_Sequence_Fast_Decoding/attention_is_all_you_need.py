"""
Title: Timeseries classification with a Transformer model
Author: [Theodoros Ntakouris](https://github.com/ntakouris)
Modified by Team A*STAR for transcription factor binding problem
Date created: 2021/06/25
Last modified: 2022/10/24
Description: Transformer model to predict the sites of CTCF transcription factor binding.
"""


"""
## Introduction

This is the Transformer architecture from
[Attention Is All You Need](https://arxiv.org/abs/1706.03762),
applied to timeseries instead of natural language.

Requires TensorFlow 2.4 or higher.

"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


"""
## Build the model

Our model processes a tensor of shape (batch size, sequence length, channels),
where sequence length is the length of the DNA sequence and channels is the
length of the one-hot encoding vector.

batch size: batch_size = 100
sequence length: max_len = 10240
channels: vocab_size = 5  # 4 base pairs (ATCG) + 1 DNase-seq data

We include residual connections, layer normalization, and dropout.
The resulting layer can be stacked multiple times.

The projection layers are implemented through `keras.layers.Conv1D`.
"""


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res


"""
The main part of our model is now complete. We can stack multiple of those
`transformer_encoder` blocks and we can also proceed to add the final
Multi-Layer Perceptron classification head. Apart from a stack of `Dense`
layers, we need to reduce the output tensor of the `TransformerEncoder` part of
our model down to a vector of features for each data point in the current
batch. A common way to achieve this is to use a pooling layer.
"""


def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

#    x = layers.GlobalAveragePooling1D(data_format="channels_last", keepdims=True)(x)

    x = layers.MaxPooling1D(pool_size=4, strides=4, padding='same')(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)

    x = layers.Conv1DTranspose(filters=5, kernel_size=7, strides=4, padding='same')(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)
