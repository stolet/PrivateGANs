import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from dataset import CelebA

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import multi_gpu_model

batch_size = 40 * 2 
num_epochs = 2 
img_height = 224
img_width = 224

celeba = CelebA(drop_features=[
    'Attractive',
    'Pale_Skin',
    'Blurry',
])

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_split = celeba.split("training", drop_zero=False)
val_split = celeba.split("validation", drop_zero=False)

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

print(train_generator)

model = keras.Sequential([
    keras.layers.Conv2D(128, kernel_size=3, strides=(1, 1), input_shape=(img_height, img_width, 3)),
    keras.layers.LeakyReLU(alpha=0.1),
    keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None),
    keras.layers.Conv2D(264, kernel_size=3, strides=(1, 1)),
    keras.layers.LeakyReLU(alpha=0.1),
    keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None),
    keras.layers.Conv2D(40, kernel_size=3, strides=(1, 1)),
    keras.layers.LeakyReLU(alpha=0.1),
    keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None),
    keras.layers.Flatten(input_shape=(26, 26, 40)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(37)
    ])


with tf.device("/cpu:0"):
    model.build()
    model.summary()

model = multi_gpu_model(model, gpus=2)

model.compile(loss='cosine_proximity',
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
    #verbose=1,
)

