import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import PIL
from tensorflow.keras import layers

from IPython import display
from generator import G_model

#tf.enable_eager_execution()

# Discriminator outputs pos values for real images and neg values for fake images
def D_model(img_height=224, img_width=224):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[img_height, img_width, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

#G = G_model()
#noise = tf.random.normal([1, 100])
#gen_img = G(noise, training=False)
#
#D = D_model()
#decision = D(gen_img)
#print(decision)
