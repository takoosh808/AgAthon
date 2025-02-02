import numpy as np
np.random.seed(1000)
import os
import matplotlib.pyplot as plt
import cv2
import keras
from PIL import Image


os.environ['KERAS_BACKEND'] = 'tensorflow'

patch_size = 8
num_of_patches = (512//patch_size)**2

image_directory = "labels/"
trainset = []
label = []

cropResidue_images = os.listdir(image_directory + "residue_background/")
for i, image_name in enumerate(cropResidue_images):
    if (image_name.split('.')[1] == 'jpg' or image_name.split('.')[1] == 'tif'):
        image = cv2.imread(image_directory + "residue_background/" + image_name)
        image = Image.fromarray(image, 'RGB')
        trainset.append(np.array(image))
        label.append(0)

sunlitShaded_images = os.listdir(image_directory + "SunlitShaded/")
for i, image_name in enumerate(sunlitShaded_images):
    if (image_name.split('.')[1] == 'jpg' or image_name.split('.')[1] == 'tif'):
        image = cv2.imread(image_directory + "SunlitShaded/" + image_name)
        image = Image.fromarray(image, 'RGB')
        trainset.append(np.array(image))
        label.append(1)


INPUT_SHAPE = (512, 512, 3)
inputs = keras.layers.Input(shape=INPUT_SHAPE)

s = keras.layers.Lambda(lambda x: x / 255)(inputs)

conv1 = keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(s)
conv1 = keras.layers.Dropout(0.1)(conv1)
conv1 = keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(conv1)
pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(conv1)
conv2 = keras.layers.Dropout(0.1)(conv2)
conv2 = keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(conv2)
pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(conv2)
conv3 = keras.layers.Dropout(0.2)(conv3)
conv3 = keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(conv3)
pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(conv3)
conv4 = keras.layers.Dropout(0.2)(conv4)
conv4 = keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(conv4)
pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = keras.layers.Conv2D(512, kernel_size=(3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(conv4)
conv5 = keras.layers.Dropout(0.3)(conv5)
conv5 = keras.layers.Conv2D(512, kernel_size=(3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(conv5)

unpooling6 = keras.layers.Conv2DTranspose(256, kernel_size = (3,3), strides = (2,2), padding = 'same')(conv5)
unpooling6 = keras.layers.concatenate([unpooling6, conv4])
conv6 = keras.layers.Conv3D(256, kernel_size = (3,3), activation ='relu', kernel_initializer='he_normal', padding = 'same')(unpooling6)
conv6 = keras.layers.Dropout(0.2)(conv6)
conv6 = keras.layers.Conv3D(256, kernel_size = (3,3), activation ='relu', kernel_initializer='he_normal', padding = 'same')(conv6)

unpooling7 = keras.layers.Conv2DTranspose(256, kernel_size = (3,3), strides = (2,2), padding = 'same')(conv6)
unpooling7 = keras.layers.concatenate([unpooling7, conv3])
conv7 = keras.layers.Conv3D(256, kernel_size = (3,3), activation ='relu', kernel_initializer='he_normal', padding = 'same')(unpooling7)
conv7 = keras.layers.Dropout(0.2)(conv7)
conv7 = keras.layers.Conv3D(256, kernel_size = (3,3), activation ='relu', kernel_initializer='he_normal', padding = 'same')(conv7)

unpooling8 = keras.layers.Conv2DTranspose(256, kernel_size = (3,3), strides = (2,2), padding = 'same')(conv7)
unpooling8 = keras.layers.concatenate([unpooling8, conv2])
conv8 = keras.layers.Conv3D(256, kernel_size = (3,3), activation ='relu', kernel_initializer='he_normal', padding = 'same')(unpooling8)
conv8 = keras.layers.Dropout(0.2)(conv8)
conv8 = keras.layers.Conv3D(256, kernel_size = (3,3), activation ='relu', kernel_initializer='he_normal', padding = 'same')(conv8)

unpooling9 = keras.layers.Conv2DTranspose(256, kernel_size = (3,3), strides = (2,2), padding = 'same')(conv8)
unpooling9 = keras.layers.concatenate([unpooling9, conv1])
conv9 = keras.layers.Conv3D(256, kernel_size = (3,3), activation ='relu', kernel_initializer='he_normal', padding = 'same')(unpooling9)
conv9 = keras.layers.Dropout(0.2)(conv9)
conv9 = keras.layers.Conv3D(256, kernel_size = (3,3), activation ='relu', kernel_initializer='he_normal', padding = 'same')(conv9)

outputs = keras.layers.Conv2D(1, (1,1), activation='sigmoid')(conv9)

model = keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
