import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow.keras.layers as layer
from tensorflow.keras import backend
# from tensorflow.keras.optimizers import Nadam
# from tensorflow.keras.callbacks import History
import keras
# import h5py
# import datetime
# import os
# import random 
# import threading

# color images
# RGB
channels = 3
height = 128
width = 128

def unet():
    input = layer.Input((channels, height, width))
    conv1 = layer.Convolution2D(32, 3, 3, border_mode='same', init='he_uniform')(input)
    conv1 = layer.normalization.BatchNormalization(mode=0, axis=1)(conv1)
    conv1 = layer.advanced_activations.ELU()(conv1)
    conv1 = layer.Convolution2D(32, 3, 3, border_mode='same', init='he_uniform')(conv1)
    conv1 = layer.normalization.BatchNormalization(mode=0, axis=1)(conv1)
    conv1 = layer.advanced_activations.ELU()(conv1)
    pool1 = layer.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layer.Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(pool1)
    conv2 = layer.normalization.BatchNormalization(mode=0, axis=1)(conv2)
    conv2 = layer.advanced_activations.ELU()(conv2)
    conv2 = layer.Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(conv2)
    conv2 = layer.normalization.BatchNormalization(mode=0, axis=1)(conv2)
    conv2 = layer.advanced_activations.ELU()(conv2)
    pool2 = layer.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layer.Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(pool2)
    conv3 = layer.normalization.BatchNormalization(mode=0, axis=1)(conv3)
    conv3 = layer.advanced_activations.ELU()(conv3)
    conv3 = layer.Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(conv3)
    conv3 = layer.normalization.BatchNormalization(mode=0, axis=1)(conv3)
    conv3 = layer.advanced_activations.ELU()(conv3)
    pool3 = layer.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layer.Convolution2D(256, 3, 3, border_mode='same', init='he_uniform')(pool3)
    conv4 = layer.normalization.BatchNormalization(mode=0, axis=1)(conv4)
    conv4 = layer.advanced_activations.ELU()(conv4)
    conv4 = layer.Convolution2D(256, 3, 3, border_mode='same', init='he_uniform')(conv4)
    conv4 = layer.normalization.BatchNormalization(mode=0, axis=1)(conv4)
    conv4 = layer.advanced_activations.ELU()(conv4)
    pool4 = layer.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = layer.Convolution2D(512, 3, 3, border_mode='same', init='he_uniform')(pool4)
    conv5 = layer.normalization.BatchNormalization(mode=0, axis=1)(conv5)
    conv5 = layer.advanced_activations.ELU()(conv5)
    conv5 = layer.Convolution2D(512, 3, 3, border_mode='same', init='he_uniform')(conv5)
    conv5 = layer.normalization.BatchNormalization(mode=0, axis=1)(conv5)
    conv5 = layer.advanced_activations.ELU()(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = layer.Convolution2D(256, 3, 3, border_mode='same', init='he_uniform')(up6)
    conv6 = layer.normalization.BatchNormalization(mode=0, axis=1)(conv6)
    conv6 = layer.advanced_activations.ELU()(conv6)
    conv6 = layer.Convolution2D(256, 3, 3, border_mode='same', init='he_uniform')(conv6)
    conv6 = layer.normalization.BatchNormalization(mode=0, axis=1)(conv6)
    conv6 = layer.advanced_activations.ELU()(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = layer.Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(up7)
    conv7 = layer.normalization.BatchNormalization(mode=0, axis=1)(conv7)
    conv7 = layer.advanced_activations.ELU()(conv7)
    conv7 = layer.Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(conv7)
    conv7 = layer.normalization.BatchNormalization(mode=0, axis=1)(conv7)
    conv7 = layer.advanced_activations.ELU()(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = layer.Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(up8)
    conv8 = layer.normalization.BatchNormalization(mode=0, axis=1)(conv8)
    conv8 = layer.advanced_activations.ELU()(conv8)
    conv8 = layer.Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(conv8)
    conv8 = layer.normalization.BatchNormalization(mode=0, axis=1)(conv8)
    conv8 = layer.advanced_activations.ELU()(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = layer.Convolution2D(32, 3, 3, border_mode='same', init='he_uniform')(up9)
    conv9 = layer.normalization.BatchNormalization(mode=0, axis=1)(conv9)
    conv9 = layer.advanced_activations.ELU()(conv9)
    conv9 = layer.Convolution2D(32, 3, 3, border_mode='same', init='he_uniform')(conv9)
    crop9 = layer.Cropping2D(cropping=((16, 16), (16, 16)))(conv9)
    conv9 = layer.normalization.BatchNormalization(mode=0, axis=1)(crop9)
    conv9 = layer.advanced_activations.ELU()(conv9)
    conv10 = Convolution2D(num_mask_channels, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=input, output=conv10)

    return model
    
