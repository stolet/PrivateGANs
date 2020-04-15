import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from dataset import CelebA
from keras.preprocessing.image import ImageDataGenerator

batch_size = 80
num_epochs = 12 

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

model = keras.Sequential([
    keras.layers.Conv2D(64, kernel_size=3, strides=(1, 1), activation='relu'),
    keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None),
    keras.layers.Conv2D(128, kernel_size=3, strides=(1, 1), activation='relu'),
    keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None),
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(37)
    ])

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
    #verbose=1
)
