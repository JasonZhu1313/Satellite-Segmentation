import pandas as pd
import os
import csv
import os
from glob import glob
from skimage.io import imread, imsave
import numpy as np

from glob import glob
import os

path_to_train = '../../data/train/'
glob_train_imgs = os.path.join(path_to_train, '*_sat.jpg')
glob_train_masks = os.path.join(path_to_train, '*_msk.png')

train_img_paths = glob(glob_train_imgs)
train_mask_paths = glob(glob_train_masks)
print(train_img_paths[:10])
print(train_mask_paths[:10])

from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray


def get_img_id(img_path):
    img_basename = os.path.basename(img_path)
    img_id = os.path.splitext(img_basename)[0][:-len('_sat')]
    return img_id


def img_gen(img_paths, img_size=(512, 512)):
    # Iterate over all image paths
    for img_path in img_paths:
        img_id = get_img_id(img_path)
        mask_path = os.path.join(path_to_train, img_id + '_msk.png')

        img = imread(img_path) / 255
        mask = rgb2gray(imread(mask_path))

        #     img = resize(img, img_size, preserve_range = True)
        #     mask = resize(mask, img_size, mode='constant', preserve_range = True)

        mask = (mask >= 0.5).astype(float)
        mask = np.reshape(mask, (512, 512, 1))
        yield img, mask


def get_non_outlier_data(train_img_paths):
    train_path_without_outlier = []
    for index, image_path in enumerate(train_img_paths):
        if index % 500 == 0:
            print(index)
        img_id = get_img_id(image_path)
        mask_path = os.path.join('../../data/train/', img_id + '_msk.png')
        mask = rgb2gray(imread(mask_path))
        if len(np.where(mask.flatten() != 0)[0]) > 800:
            train_path_without_outlier.append(image_path)
    return train_path_without_outlier


train_img_paths = get_non_outlier_data(train_img_paths)
import numpy as np

# Keras takes its input in batches
# (i.e. a batch size of 32 would correspond to 32 images and 32 masks from the generator)

# The generator should run forever
import numpy as np

# Keras takes its input in batches
# (i.e. a batch size of 32 would correspond to 32 images and 32 masks from the generator)
# The generator should run forever

# Keras takes its input in batches
# (i.e. a batch size of 32 would correspond to 32 images and 32 masks from the generator)
# The generator should run forever
def image_batch_generator(img_paths, batchsize=32):
    while True:
        ig = img_gen(img_paths)
        batch_img, batch_mask = [], []

        for img, mask in ig:
            # Add the image and mask to the batch
            batch_img.append(img)
            batch_mask.append(mask)
            # If we've reached our batchsize, yield the batch and reset
            if len(batch_img) == batchsize:
                yield np.stack(batch_img, axis=0), np.stack(batch_mask, axis=0)
                batch_img, batch_mask = [], []

        # If we have an nonempty batch left, yield it out and reset
        if len(batch_img) != 0:
            yield np.stack(batch_img, axis=0), np.stack(batch_mask, axis=0)
            batch_img, batch_mask = [], []


from sklearn.model_selection import train_test_split

BATCHSIZE = 1

# Split the data into a train and validation set
train_img_paths, val_img_paths = train_test_split(train_img_paths, test_size=0.1,random_state=42)
print(len(train_img_paths))
print(len(val_img_paths))

# Create the train and validation generators
traingen = image_batch_generator(train_img_paths, batchsize=BATCHSIZE)
valgen = image_batch_generator(val_img_paths, batchsize=BATCHSIZE)
print(next(traingen)[0].shape)
def calc_steps(data_len, batchsize):
    return (data_len + batchsize - 1) // batchsize

# Calculate the steps per epoch
train_steps = calc_steps(len(train_img_paths), BATCHSIZE)
val_steps = calc_steps(len(val_img_paths), BATCHSIZE)
print("train steps {}".format(train_steps))
print("val steps {}".format(val_steps))

import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import *
from keras import backend as K

import tensorflow as tf

# Build U-Net model
inputs = Input((512, 512, 3))
s = inputs

c0 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(inputs)
c0 = Dropout(0.1)(c0)
c0 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c0)
p0 = MaxPooling2D((2, 2))(c0)

c1 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p0)
c1 = Dropout(0.1)(c1)
c1 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
p1 = MaxPooling2D((2, 2))(c1)

c2 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
c2 = Dropout(0.1)(c2)
c2 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
p2 = MaxPooling2D((2, 2))(c2)

c3 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
c3 = Dropout(0.2)(c3)
c3 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
p3 = MaxPooling2D((2, 2))(c3)

c4 = Conv2D(512, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
c4 = Dropout(0.3)(c4)
c4 = Conv2D(512, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
p4 = MaxPooling2D(pool_size=(2, 2))(c4)

c5 = Conv2D(1024, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
c5 = Dropout(0.5)(c5)
c5 = Conv2D(1024, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(512, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
c6 = Dropout(0.3)(c6)
c6 = Conv2D(512, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
c7 = Dropout(0.2)(c7)
c7 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
c8 = Dropout(0.1)(c8)
c8 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = concatenate([u9, c1])
c9 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
c9 = Dropout(0.1)(c9)
c9 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

u10 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c9)
u10 = concatenate([u10, c0], axis=3)
c10 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u10)
c10 = Dropout(0.1)(c10)
c10 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c10)

outputs = Conv2D(1, (1, 1), activation='sigmoid')(c10)

import keras.backend as K
from keras.optimizers import Adam
from keras.losses import binary_crossentropy

smooth = 1e-9


# This is the competition metric implemented using Keras
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * (K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


# We'll construct a Keras Loss that incorporates the DICE score
def dice_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1. - (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)


def bce_dice_loss(y_true, y_pred):
    return 0.3 * binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def ln_dice(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * (K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return K.log(score)


def new_bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - ln_dice(y_true, y_pred)


model = Model(inputs=[inputs], outputs=[outputs])
model.compile(Adam(1e-4), loss=bce_dice_loss, metrics=[dice_coef])
model.summary()
MODEL_NAME = 'nonblack_13layer-{epoch:02d}-{val_loss:.4f}-id7.h5'
import math
checkpointer = ModelCheckpoint(MODEL_NAME, verbose=1,monitor='val_loss', save_best_only=True)
def step_decay(epoch):
   initial_lrate = 1e-4
   drop = 0.5
   epochs_drop = 15.0
   lrate = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
   return lrate
lrate = LearningRateScheduler(step_decay)
tensorboard = TensorBoard(log_dir='./logs',
                                write_graph=False, #This eats a lot of space. Enable with caution!
                                #histogram_freq = 1,
                                write_images=True,
                                batch_size = 1,
                                write_grads=True)
EPOCH_NUM = 70
# Train the model
history = model.fit_generator(
    traingen,
    steps_per_epoch=train_steps,
    epochs=EPOCH_NUM, # Change this to a larger number to train for longer
    validation_data=valgen,
    validation_steps=val_steps,
    verbose=1,
    max_queue_size=5,  # Change this number based on memory restrictions
    callbacks = [checkpointer,lrate]
)