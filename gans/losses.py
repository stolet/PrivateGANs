import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.backend import binary_crossentropy as cross_entropy_vector
from tensorflow.nn import sigmoid_cross_entropy_with_logits as scel


#def D_loss(real_out, fake_out):
#    real_loss = cross_entropy(tf.ones_like(real_out), real_out)
#    fake_loss = cross_entropy(tf.zeros_like(fake_out), fake_out)
#    total_loss = real_loss + fake_loss
#    return total_loss
#
#def D_loss_dp(real_out, fake_out):
#    real_loss = cross_entropy_vector(tf.ones_like(real_out), real_out, from_logits=True)
#    fake_loss = cross_entropy_vector(tf.zeros_like(fake_out), fake_out, from_logits=True)
#    total_loss = real_loss + fake_loss
#    return total_loss
#
#
#def G_loss(fake_output):
#    return cross_entropy(tf.ones_like(fake_output), fake_output)


def D_loss(fake_out, fake_logits, real_out, real_logits, smooth_factor=0.1):
    loss_real = scel(logits=real_logits, labels=tf.ones_like(real_out) * (1 - smooth_factor))
    loss_real = tf.reduce_mean(loss_real)

    loss_fake = scel(logits=fake_logits, labels=tf.zeros_like(fake_out))
    loss_fake = tf.reduce_mean(loss_fake)
    return loss_real + loss_fake

def D_loss_dp(fake_out, fake_logits, real_out, real_logits, smooth_factor=0.1):
    loss_real = scel(logits=real_logits, labels=tf.ones_like(real_out) * (1 - smooth_factor))
    loss_fake = scel(logits=fake_logits, labels=tf.ones_like(fake_out))
    return loss_real + loss_fake

def G_loss(fake_out, fake_logits):
    loss = tf.reduce_mean(scel(logits=fake_logits, labels=tf.ones_like(fake_out)))
    return loss 
