import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import PIL
from tensorflow.keras import layers

from IPython import display

tf.enable_eager_execution()

#def G_model():
#    model = tf.keras.Sequential()
#    model.add(layers.Dense(4*4*256, use_bias=False, input_shape=(100,)))
#    model.add(layers.BatchNormalization())
#    model.add(layers.LeakyReLU())
#
#    model.add(layers.Reshape((4, 4, 256)))
#    assert model.output_shape == (None, 4, 4, 256) # Note: None is the batch size
#
#    
#    model.add(layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='same', use_bias=False))
#    assert model.output_shape == (None, 4, 4, 64)
#    model.add(layers.BatchNormalization())
#    model.add(layers.LeakyReLU())
#    
#    
#    model.add(layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', use_bias=False))
#    assert model.output_shape == (None, 8, 8, 128)
#    model.add(layers.BatchNormalization())
#    model.add(layers.LeakyReLU())
#
#    
#    model.add(layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', use_bias=False))
#    assert model.output_shape == (None, 16, 16, 128)
#    model.add(layers.BatchNormalization())
#    model.add(layers.LeakyReLU())
#    
#    
#    model.add(layers.Conv2DTranspose(264, (3, 3), strides=(2, 2), padding='same', use_bias=False))
#    assert model.output_shape == (None, 32, 32, 264)
#    model.add(layers.BatchNormalization())
#    model.add(layers.LeakyReLU())
# 
#    
#    model.add(layers.Conv2DTranspose(264, (3, 3), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
#    assert model.output_shape == (None, 64, 64, 264)
#    model.add(layers.BatchNormalization())
#    model.add(layers.LeakyReLU())
#
#    
#    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', use_bias=False))
#    assert model.output_shape == (None, 32, 32, 128)
#    model.add(layers.BatchNormalization())
#    model.add(layers.LeakyReLU())
# 
#    
#    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
#    assert model.output_shape == (None, 16, 16, 128)
#    model.add(layers.BatchNormalization())
#    model.add(layers.LeakyReLU())
#
#
#    model.add(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', use_bias=False))
#    assert model.output_shape == (None, 32, 32, 64)
#    model.add(layers.BatchNormalization())
#    model.add(layers.LeakyReLU())
# 
#    
#    model.add(layers.Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
#    assert model.output_shape == (None, 64, 64, 3)
#    model.add(layers.BatchNormalization())
#    model.add(layers.LeakyReLU())
#    
#    return model


def Generator(noise, out_channel_dim, is_train=True, alpha=0.2, keep_prob=0.5):
    with tf.variable_scope('Generator', reuse=(not is_train)):

        # Fully connected layer 1
        fc = layers.Dense(4 * 4 * 1024, use_bias=False, input_shape=(100,))(noise)
        fc = tf.reshape(fc, (-1, 4, 4, 1024))
        
        bn1 = layers.BatchNormalization()(fc, training=is_train)
        lrelu1 = tf.maximum(alpha * bn1, bn1)
        drop1 = layers.Dropout(keep_prob)(lrelu1, training=is_train)

        # Transpose Conv 1
        transconv1 = layers.Conv2DTranspose(512, 4, 1, 'valid', use_bias=False)(drop1)
        
        bn2 = layers.BatchNormalization()(transconv1, training=is_train)
        lrelu2 = tf.maximum(alpha * bn2, bn2)
        drop2 = layers.Dropout(keep_prob)(lrelu2, training=is_train)

        # Transpose Conv 2
        transconv2 = layers.Conv2DTranspose(256, 5, 2, 'same', use_bias=False)(drop2)
        
        bn3 = layers.BatchNormalization()(transconv2, training=is_train)
        lrelu3 = tf.maximum(alpha * bn3, bn3)
        drop3 = layers.Dropout(keep_prob)(lrelu3, training=is_train)

        # Output
        out = layers.Conv2DTranspose(out_channel_dim, 5, 2, 'same', activation='tanh')(drop3)
        return out


noise = tf.random.normal([1, 100])
out_channel_dim = 5
generated_image = Generator(noise=noise, out_channel_dim=out_channel_dim)

#noise = tf.random.normal([1, 100])
#generated_image = G(noise, training=False)
#
plt.imshow(generated_image[0, :, :, 0], cmap = 'viridis', interpolation='bicubic')
plt.savefig("stuff.jpg")
