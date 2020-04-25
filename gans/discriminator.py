import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import PIL
from tensorflow.keras import layers
from tensorflow.contrib.layers import xavier_initializer
from IPython import display
from generator import Generator

tf.enable_eager_execution()

def Discriminator(input_shape, reuse=False, alpha=0.2, keep_prob=0.5):
    
    # Input
    inputs = keras.Input(shape=input_shape)

    # Conv layer 1 
    conv1 = layers.Conv2D(64, 5, 2, padding='same', kernel_initializer=xavier_initializer())(inputs)
    lrelu1 = tf.maximum(alpha * conv1, conv1)
    drop1 = layers.Dropout(keep_prob)(lrelu1)

    # Conv layer 2
    conv2 = layers.Conv2D(128, 5, 2, 'same', use_bias=False)(drop1)
    bn1 = layers.BatchNormalization()(conv2)
    lrelu2 = tf.maximum(alpha * bn1, bn1)
    drop2 = layers.Dropout(keep_prob)(lrelu2)

    # Conv layer 3
    conv3 = layers.Conv2D(256, 5, 2, 'same', use_bias=False)(drop2)
    bn3 = layers.BatchNormalization()(conv3)
    lrelu3 = tf.maximum(alpha * bn3, bn3)
    drop3 = layers.Dropout(keep_prob)(lrelu3)

    # Output
    flat = tf.reshape(drop3, (-1, 4 * 4 * 256))
    logits = layers.Dense(1)(flat)
    out = keras.activations.sigmoid(logits)

    # Create model
    model = keras.Model(inputs=inputs, outputs={"out": out, "logits": logits})
    return model

#OUT_CHANNEL_DIM = 5
#G_INPUT_SHAPE = (100)
#
#noise = tf.random.normal([1, 100])
#G = Generator(G_INPUT_SHAPE, OUT_CHANNEL_DIM)
#gen_img = G(noise)
#
#D_INPUT_SHAPE = (28, 28, OUT_CHANNEL_DIM)
#D = Discriminator(D_INPUT_SHAPE)
#decision = D(gen_img)
#print(decision)
