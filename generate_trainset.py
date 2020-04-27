import tensorflow as tf
import glob
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import PIL

from tensorflow.keras.preprocessing.image import save_img
from PIL import Image
from tensorflow import keras
from gans.generator import Generator

tf.enable_eager_execution()

NUM_EXAMPLES_TO_GENERATE = 2000
IMG_HEIGHT               = 28
IMG_WIDTH                = 28
NOISE_DIM                = 100
CHECKPOINT_DIR           = "./gans/celeba_o_dp_13/"
GENERATED_DATA_DIR       = "./synthetic_data_dp_13/celeba_old/"

OUT_CHANNEL_DIM          = 3
G_INPUT_SHAPE            = (NOISE_DIM,)

G = Generator(G_INPUT_SHAPE, OUT_CHANNEL_DIM)
checkpoint = tf.train.Checkpoint(Generator=G)
checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR)).expect_partial()


noise = np.random.uniform(-1, 1, size=(NUM_EXAMPLES_TO_GENERATE, NOISE_DIM))
new_images = G(noise, training=False)

for i in range(NUM_EXAMPLES_TO_GENERATE):
    array = new_images[i,:]
    save_img(GENERATED_DATA_DIR + str(i) + ".jpeg", array)
