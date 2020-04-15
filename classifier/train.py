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

train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()

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
