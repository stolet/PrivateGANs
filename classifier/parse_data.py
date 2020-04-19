import numpy as np
from dataset import CelebA
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications.vgg16 import VGG16
import pandas as pd
import shutil

celeba = CelebA(drop_features=['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie'])

train_split = celeba.split("training", drop_zero=False)
val_split = celeba.split("validation", drop_zero=False)
test_split = celeba.split("test", drop_zero=False)

for index, row in train_split.iterrows():
    young = row["Young"]
    image_id = row["image_id"]
    if young == 1:
        shutil.copy("/home/stolet/Documents/PrivateGANs/data/img_align_celeba/img_align_celeba/" + image_id, "/home/stolet/Documents/PrivateGANs/data/celeba_young/" + image_id)
    else:
        shutil.copy("/home/stolet/Documents/PrivateGANs/data/img_align_celeba/img_align_celeba/" + image_id, "/home/stolet/Documents/PrivateGANs/data/celeba_old/" + image_id)

for index, row in val_split.iterrows():
    young = row["Young"]
    image_id = row["image_id"]
    if young == 1:
        shutil.copy("/home/stolet/Documents/PrivateGANs/data/img_align_celeba/img_align_celeba/" + image_id, "/home/stolet/Documents/PrivateGANs/data/celeba_young/" + image_id)
    else:
        shutil.copy("/home/stolet/Documents/PrivateGANs/data/img_align_celeba/img_align_celeba/" + image_id, "/home/stolet/Documents/PrivateGANs/data/celeba_old/" + image_id)

for index, row in val_split.iterrows():
    young = row["Young"]
    image_id = row["image_id"]
    if young == 1:
        shutil.copy("/home/stolet/Documents/PrivateGANs/data/img_align_celeba/img_align_celeba/" + image_id, "/home/stolet/Documents/PrivateGANs/data/celeba_young/" + image_id)
    else:
        shutil.copy("/home/stolet/Documents/PrivateGANs/data/img_align_celeba/img_align_celeba/" + image_id, "/home/stolet/Documents/PrivateGANs/data/celeba_old/" + image_id)

