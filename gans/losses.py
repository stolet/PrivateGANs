import tensorflow as tf
import tensorflow.keras as keras

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def D_loss(real_out, fake_out):
    real_loss = cross_entropy(tf.ones_like(real_out), real_out)
    fake_loss = cross_entropy(tf.zeros_like(fake_out), fake_out)
    total_loss = real_loss + fake_loss
    return total_loss

def G_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
