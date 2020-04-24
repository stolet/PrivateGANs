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
NUM_EPOCHS           = 3000
IMG_HEIGHT           = 64
IMG_WIDTH            = 64
NOISE_DIM            = 100
YOUNG                = True
LEARNING_RATE_G      = 1e-4
LEARNING_RATE_D      = 1e-4

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

G_optimizer = keras.optimizers.Adam(LEARNING_RATE_G)
D_optimizer = keras.optimizers.Adam(LEARNING_RATE_D)

G = G_model()
D = D_model(img_height=IMG_HEIGHT, img_width=IMG_WIDTH)

checkpoint_dir = './checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(G_optimizer=G_optimizer,
                                 D_optimizer=D_optimizer,
                                 Generator=G,
                                 Discriminator=D)



checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

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
        
        # Break loop when one epoch is finished
        if i >= len(train_generator) / BATCH_SIZE:
            break
    
        i += 1
        
    print ('Saving: Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    
    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

# Generate after the final epoch
display.clear_output(wait=True)
generate_and_save_images(G, NUM_EPOCHS, seed)

