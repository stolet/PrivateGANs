import tensorflow as tf

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time
from tensorflow_privacy import DPAdamGaussianOptimizer

from tensorflow import keras
from tensorflow.keras import layers
from dataset import CelebA
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from IPython import display

from generator import Generator
from discriminator import Discriminator
from losses import D_loss, G_loss

tf.enable_eager_execution()

# General params
NUM_EXAMPLES_TO_GENERATE = 16
BATCH_SIZE               = 80
NUM_EPOCHS               = 10
IMG_HEIGHT               = 28
IMG_WIDTH                = 28
NOISE_DIM                = 100
YOUNG                    = True
NOISE_SHAPE              = [BATCH_SIZE, NOISE_DIM]

# Generator hyperparams
OUT_CHANNEL_DIM          = 3
G_INPUT_SHAPE            = (NOISE_DIM)

# Discriminator hyperparams
D_INPUT_SHAPE            = (IMG_HEIGHT, IMG_WIDTH, OUT_CHANNEL_DIM)

# Optimizer params
LEARNING_RATE_G          = 0.00025
LEARNING_RATE_D          = 0.00025
BETA1                    = 0.45

# Differential privacy hyperparams
DP_ON                    = False
L2_CLIP                  = 1.5
NOISE_MULT               = 1
MICROBATCHES             = BATCH_SIZE


celeba = celeba = CelebA(drop_features=['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie'])

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_split = celeba.split("training", drop_zero=False)
val_split = celeba.split("validation", drop_zero=False)

if YOUNG:
    is_young_t = train_split["Young"]==1
    is_young_v = val_split["Young"]==1
    train_split = train_split[is_young_t]
    val_split = val_split[is_young_v] 
else :
    is_old_t = train_split["Young"]==0
    is_old_v = val_split["Young"]==0
    train_split = train_split[is_old_t]
    val_split = val_split[is_old_v]

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_split,
    directory=celeba.images_folder,
    x_col='image_id',
    y_col=celeba.features_name,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="raw",
)

valid_generator = val_datagen.flow_from_dataframe(
    dataframe=val_split,
    directory=celeba.images_folder,
    x_col='image_id',
    y_col=celeba.features_name,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="raw",
)

G_optimizer = tf.train.AdamOptimizer(LEARNING_RATE_G, beta1=BETA1)
D_optimizer = tf.train.AdamOptimizer(LEARNING_RATE_D, beta1=BETA1)
if DP_ON:
    D_optimizer = DPAdamGaussianOptimizer(learning_rate=LEARNING_RATE_D,
                                          l2_norm_clip=L2_CLIP,
                                          noise_multiplier=NOISE_MULT,
                                          num_microbatches=MICROBATCHES)

G = Generator(G_INPUT_SHAPE, OUT_CHANNEL_DIM) 
D = Discriminator(D_INPUT_SHAPE)

checkpoint_dir = './checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(G_optimizer=G_optimizer,
                                 D_optimizer=D_optimizer,
                                 Generator=G,
                                 Discriminator=D)

#checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

def generate_and_save_images(model, epoch, test_input):
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i,:], interpolation="bilinear")
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

seed = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIM])

for epoch in range(NUM_EPOCHS):

    start = time.time()
    i = 0
    for image_batch in train_generator:
        imgs = image_batch[0]
        labels = image_batch[1]
        noise = tf.random.normal(NOISE_SHAPE)

        with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:
            gen_imgs = G(noise, training=True)

            real_out = D(imgs, training=True)
            real_preds = real_out["out"]
            real_logits = real_out["logits"]

            fake_out = D(gen_imgs, training=True)
            fake_preds = fake_out["out"]
            fake_logits = fake_out["logits"]

            g_loss = G_loss(fake_preds, fake_logits)
            if DP_ON:
                d_loss = D_loss_dp(fake_preds, fake_logits, real_preds, real_logits)
            else:
                d_loss = D_loss(fake_preds, fake_logits, real_preds, real_logits)

        G_grads = G_tape.gradient(g_loss, G.trainable_variables)
        D_grads = D_tape.gradient(d_loss, D.trainable_variables)

        G_optimizer.apply_gradients(zip(G_grads, G.trainable_variables))
        D_optimizer.apply_gradients(zip(D_grads, D.trainable_variables))
        
        # Break loop when one epoch is finished
        if i >= len(train_generator):
            break
    
        i += 1
    
    print ('Saving: Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    
    # Save the model every 15 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

# Generate after the final epoch
display.clear_output(wait=True)
generate_and_save_images(G, NUM_EPOCHS, seed)

if DP_ON:
    sampling_prob = 1 / len(train_generator)
    steps = NUM_EPOCHS * len(train_generator) * BATCH_SIZE
    
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    rdp = compute_rdp(q=sampling_prob, noise_multiplier=NOISE_MULT, steps=steps, orders=orders)
    epsilon = get_privacy_spent(orders, rdp, targe_delta=1e-5)[0]
    print("Epsilon: " + str(epsilon))
