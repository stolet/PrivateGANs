import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow.keras as keras
from tensorflow.keras import layers

from IPython import display

#tf.enable_eager_execution()

def Generator(input_shape, out_channel_dim, is_train=True, alpha=0.2, keep_prob=0.5):
    
    # Input
    inputs = keras.Input(shape=input_shape)

    # Fully connected layer 1
    fc = layers.Dense(4 * 4 * 1024, use_bias=False, input_shape=input_shape)(inputs)
    fc = tf.reshape(fc, (-1, 4, 4, 1024))
    
    bn1 = layers.BatchNormalization()(fc)
    lrelu1 = tf.maximum(alpha * bn1, bn1)
    drop1 = layers.Dropout(keep_prob)(lrelu1)

    # Transpose Conv 1
    transconv1 = layers.Conv2DTranspose(512, 4, 1, 'valid', use_bias=False)(drop1)
    
    bn2 = layers.BatchNormalization()(transconv1)
    lrelu2 = tf.maximum(alpha * bn2, bn2)
    drop2 = layers.Dropout(keep_prob)(lrelu2)

    # Transpose Conv 2
    transconv2 = layers.Conv2DTranspose(256, 5, 2, 'same', use_bias=False)(drop2)
    
    bn3 = layers.BatchNormalization()(transconv2)
    lrelu3 = tf.maximum(alpha * bn3, bn3)
    drop3 = layers.Dropout(keep_prob)(lrelu3)

    # Output
    out = layers.Conv2DTranspose(out_channel_dim, 5, 2, 'same', activation='tanh')(drop3)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=out)
    return model


#noise = tf.random.normal([1, 100])
#out_channel_dim = 5
#generated_image = Generator(noise=noise, out_channel_dim=out_channel_dim)
#plt.imshow(generated_image[0, :, :, 0], cmap = 'viridis', interpolation='bicubic')
#plt.savefig("stuff.jpg")
