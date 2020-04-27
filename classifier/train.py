import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from dataset import CelebA

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

batch_size = 20 * 2 
num_epochs = 5 
img_height = 224
img_width = 224

#celeba = CelebA(drop_features=[
#    'Attractive',
#    'Pale_Skin',
#    'Blurry',
#])

celeba = CelebA(drop_features=['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie'])

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_split = celeba.split("training", drop_zero=False)
val_split = celeba.split("validation", drop_zero=False)

print(train_split)
#synthetic_split
#synthetic_dp_13_split
#synthetic_dp_8_split

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_split,
    directory=celeba.images_folder,
    x_col='image_id',
    y_col=celeba.features_name,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode="raw",
)

valid_generator = val_datagen.flow_from_dataframe(
    dataframe=val_split,
    directory=celeba.images_folder,
    x_col='image_id',
    y_col=celeba.features_name,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode="raw",
)

#model = keras.Sequential([
#    keras.layers.Conv2D(128, kernel_size=3, strides=(1, 1), input_shape=(img_height, img_width, 3)),
#    keras.layers.LeakyReLU(alpha=0.1),
#    keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None),
#    keras.layers.Conv2D(264, kernel_size=3, strides=(1, 1)),
#    keras.layers.LeakyReLU(alpha=0.1),
#    keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None),
#    keras.layers.Conv2D(40, kernel_size=3, strides=(1, 1)),
#    keras.layers.LeakyReLU(alpha=0.1),
#    keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None),
#    keras.layers.Flatten(input_shape=(26, 26, 40)),
#    keras.layers.Dense(128, activation='relu'),
#    keras.layers.Dense(37)
#    ])

model = keras.Sequential()
model.add(MobileNetV2(None))
model.add(keras.layers.Dense(1))

with tf.device("/cpu:0"):
    model.build()
    model.summary()

#model = multi_gpu_model(model, gpus=2)

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])

history = model.fit_generator(
    train_generator,
    epochs=num_epochs,
    steps_per_epoch=len(train_generator),
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
    max_queue_size=1,
    shuffle=True,
    verbose=1,
)

model.save_weights("models/weights.h5")

