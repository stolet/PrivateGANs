import tensorflow as tf

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time

from tensorflow import keras
from tensorflow.keras import layers
from dataset import CelebA
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from IPython import display

from generator import G_model
from discriminator import D_model
from losses import D_loss, G_loss

tf.enable_eager_execution()

NUM_EXAMPLES_TO_GENERATE = 16
BATCH_SIZE           = 80
NUM_EPOCHS           = 1
IMG_HEIGHT           = 224
IMG_WIDTH            = 224
NOISE_DIM            = 100

celeba = celeba = CelebA(drop_features=['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie'])

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_split = celeba.split("training", drop_zero=False)
val_split = celeba.split("validation", drop_zero=False)

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

G_optimizer = keras.optimizers.Adam(1e-4)
D_optimizer = keras.optimizers.Adam(1e-4)

G = G_model()
D = D_model()

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(G_optimizer=G_optimizer,
                                 D_optimizer=D_optimizer,
                                 Generator=G,
                                 Discriminator=D)

def generate_and_save_images(model, epoch, test_input):
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

print(len(train_generator))
seed = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIM])

for epoch in range(NUM_EPOCHS):

    start = time.time()
    i = 0
    for image_batch in train_generator:
        imgs = image_batch[0]
        labels = image_batch[1]
        noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
        
        with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:
            gen_imgs = G(noise, training=True)

            real_preds = D(imgs, training=True)
            fake_preds = D(gen_imgs, training=True)

            g_loss = G_loss(fake_preds)
            d_loss = D_loss(real_preds, fake_preds)

        G_grads = G_tape.gradient(g_loss, G.trainable_variables)
        D_grads = D_tape.gradient(d_loss, D.trainable_variables)

        G_optimizer.apply_gradients(zip(G_grads, G.trainable_variables))
        D_optimizer.apply_gradients(zip(D_grads, D.trainable_variables))
    
        i += 1
        if i % 100 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            print(i)
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Produce images for the GIF as we go
    #display.clear_output(wait=True)
    #generate_and_save_images(G, epoch + 1, seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

# Generate after the final epoch
tf.enable_eager_execution()
display.clear_output(wait=True)
generate_and_save_images(G, NUM_EPOCHS, seed)

