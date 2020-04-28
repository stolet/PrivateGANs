import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from dataset import CelebA
from synthetic_dataset import SyntheticCelebA

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
tf.enable_eager_execution()

batch_size = 64 
num_epochs = 40 
img_height = 32
img_width = 32

celeba = CelebA(drop_features=['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie'])


train_datagen = ImageDataGenerator(rescale=1./255, samplewise_center=True, samplewise_std_normalization=True)
val_datagen = ImageDataGenerator(rescale=1./255, samplewise_center=True, samplewise_std_normalization=True)
test_datagen = ImageDataGenerator(rescale=1./255, samplewise_center=True, samplewise_std_normalization=True)

def balance_dataset(n, data):
    n_old = 0
    n_young = 0
    young_data = pd.DataFrame({"image_id": [], "Young": []}, columns = ["image_id", "Young"])
    old_data = pd.DataFrame({"image_id": [], "Young": []}, columns = ["image_id", "Young"])
    
    for index, elt in data.iterrows():
        if elt["Young"] == 1 and n_young < n / 2:
            n_young += 1
            young_data = young_data.append(elt)
        elif elt["Young"] == 0 and n_old < n / 2:
            n_old += 1
            old_data = old_data.append(elt)

        if n_young >= n / 2 and n_old >= n / 2:
            break

    df = pd.concat([young_data, old_data])
    df = df.sample(frac=1, axis=0)
    return df

train_split = celeba.split("training", drop_zero=False)
val_split = celeba.split("validation", drop_zero=False)
test_split = celeba.split("test", drop_zero=False)

train_split = balance_dataset(1000, train_split)
val_split = balance_dataset(2000, val_split)
test_split = balance_dataset(5000, test_split)


synthetic_split = SyntheticCelebA(main_folder='../synthetic_data/')
synthetic_split = synthetic_split.dataset
#synthetic_dp_13_split = SyntheticCelebA(main_folder='../synthetic_data_dp_13/')
#synthetic_dp_8_split = SyntheticCelebA(main_folder='../synthetic_data_dp_8/') 

train_generator = train_datagen.flow_from_dataframe(
    dataframe=synthetic_split,
    directory='../synthetic_data/',
    x_col='image_id',
    y_col=celeba.features_name,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="raw",
)

valid_generator = val_datagen.flow_from_dataframe(
    dataframe=val_split,
    directory=celeba.images_folder,
    x_col='image_id',
    y_col=celeba.features_name,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="raw",
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_split,
    directory=celeba.images_folder,
    x_col='image_id',
    y_col=celeba.features_name,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="raw",
)

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
    keras.layers.Dense(1, activation='sigmoid')
    ])

with tf.device("/cpu:0"):
    model.build()
    model.summary()

adam = keras.optimizers.Adam(lr=0.001)
loss = keras.losses.BinaryCrossentropy()
model.compile(loss=loss,
              optimizer=adam,
              metrics=['accuracy'])

early_stoppping_callback = EarlyStopping(monitor='val_acc', patience=3, min_delta=0.01)
history = model.fit_generator(
    train_generator,
    epochs=num_epochs,
    steps_per_epoch=len(train_generator),
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
    max_queue_size=1,
    shuffle=True,
    verbose=1,
    #callbacks=[early_stoppping_callback],
)

model.save_weights("models/weights.h5")

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

