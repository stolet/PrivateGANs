import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.backend import binary_crossentropy as cross_entropy_vector

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def D_loss(real_out, fake_out):
    real_loss = cross_entropy(tf.ones_like(real_out), real_out)
    fake_loss = cross_entropy(tf.zeros_like(fake_out), fake_out)
    total_loss = real_loss + fake_loss
    return total_loss

def D_loss_dp(real_out, fake_out):
    real_loss = cross_entropy_vector(tf.ones_like(real_out), real_out, from_logits=True)
    fake_loss = cross_entropy_vector(tf.zeros_like(fake_out), fake_out, from_logits=True)
    total_loss = real_loss + fake_loss
    return total_loss


def G_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
