import numpy as np
np.random.seed(1000)
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import keras
from PIL import Image

os.environ['KERAS_BACKEND'] = 'tensorflow'

patch_size = 8
num_of_patches = (512//patch_size)**2

# image_directory = "labels/"
# dataset = []
# label = []

# cropResidue_images = os.listdir(image_directory + "residue_background/")
# for i, image_name in enumerate(cropResidue_images):
#     if (image_name.split('.')[1] == 'jpg'):
#         image = cv2.imread(image_directory + "residue_background/" + image_name)
#         image = Image.fromarray(image, 'RGB')
#         dataset.append(np.array(image))
#         label.append(0)

# for i, image_name in enumerate(cropResidue_images):
#     if(image_name.split('.')[1]=='tif'):
#         image = cv2.imread(image_directory + "residue_background/" + image_name)
#         image = Image.fromarray(image, 'RGB')
#         dataset.append(np.array(image))
#         label.append(1)


# def resize(input_image, input_mask):
#     input_image = tf.image.resize(input_image,(128,128),method="nearest")
#     input_mask = tf.image.resize(input_mask,(128,128),method="nearest")
#     return input_image,input_mask

# def augment(input_image, input_mask):
#     input_image = tf.image.flip_up_down(input_image)
#     input_mask = tf.image.flip_up_down(input_mask)
#     return input_image, input_mask

# def normalize(input_image, input_mask):
#     input_image = tf.cast(input_image,tf.float32)/255.0
#     input_mask -= 1
#     return input_image, input_mask

# def load_image_train(datapoint):
#    input_image = datapoint["image"]
#    input_mask = datapoint["segmentation_mask"]
#    input_image, input_mask = resize(input_image, input_mask)
#    input_image, input_mask = augment(input_image, input_mask)
#    input_image, input_mask = normalize(input_image, input_mask)

#    return input_image, input_mask

# def load_image_test(datapoint):
#    input_image = datapoint["image"]
#    input_mask = datapoint["segmentation_mask"]
#    input_image, input_mask = resize(input_image, input_mask)
#    input_image, input_mask = normalize(input_image, input_mask)

#    return input_image, input_mask

# train_dataset = dataset["train"].map(load_image_train, num_parallel_calls = tf.data.AUTOTUNE)
# test_dataset = dataset["test"].map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)


IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3
inputs = keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

s = keras.layers.Lambda(lambda x: x / 255)(inputs)

c1 = keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(s)
c1 = keras.layers.Dropout(0.1)(c1)
c1 = keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(c1)
p1 = keras.layers.MaxPooling2D((2, 2))(c1)


c2 = keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(p1)
c2 = keras.layers.Dropout(0.1)(c2)
c2 = keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(c2)
p2 = keras.layers.MaxPooling2D((2, 2))(c2)


c3 = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(p2)
c3 = keras.layers.Dropout(0.2)(c3)
c3 = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(c3)
p3 = keras.layers.MaxPooling2D((2, 2))(c3)


c4 = keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(p3)
c4 = keras.layers.Dropout(0.2)(c4)
c4 = keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(c4)
p4 = keras.layers.MaxPooling2D((2, 2))(c4)

c5 = keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(p4)
c5 = keras.layers.Dropout(0.3)(c5)
c5 = keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(c5)

u6 = keras.layers.Conv2DTranspose(128, (2,2), strides = (2,2), padding = 'same')(c5)
u6 = keras.layers.Concatenate()([u6,c4])
c6 = keras.layers.Conv2D(128, (3,3), activation ='relu', kernel_initializer='he_normal', padding = 'same')(u6)
c6 = keras.layers.Dropout(0.2)(c6)
c6 = keras.layers.Conv2D(128, (3,3), activation ='relu', kernel_initializer='he_normal', padding = 'same')(c6)

u7 = keras.layers.Conv2DTranspose(64, (2,2), strides = (2,2), padding = 'same')(c6)
u7 = keras.layers.Concatenate()([u7, c3])
c7 = keras.layers.Conv2D(64, (3,3), activation ='relu', kernel_initializer='he_normal', padding = 'same')(u7)
c7 = keras.layers.Dropout(0.2)(c7)
c7 = keras.layers.Conv2D(64, (3,3), activation ='relu', kernel_initializer='he_normal', padding = 'same')(c7)

u8 = keras.layers.Conv2DTranspose(32, (2,2), strides = (2,2), padding = 'same')(c7)
u8 = keras.layers.Concatenate()([u8, c2])
c8 = keras.layers.Conv2D(32, (3,3), activation ='relu', kernel_initializer='he_normal', padding = 'same')(u8)
c8 = keras.layers.Dropout(0.1)(c8)
c8 = keras.layers.Conv2D(32, (3,3), activation ='relu', kernel_initializer='he_normal', padding = 'same')(c8)

u9 = keras.layers.Conv2DTranspose(16, (2,2), strides = (2,2), padding = 'same')(c8)
u9 = keras.layers.Concatenate()([u9, c1])
c9 = keras.layers.Conv2D(16, (3,3), activation ='relu', kernel_initializer='he_normal', padding = 'same')(u9)
c9 = keras.layers.Dropout(0.1)(c9)
c9 = keras.layers.Conv2D(16, (3,3), activation ='relu', kernel_initializer='he_normal', padding = 'same')(c9)

outputs = keras.layers.Conv2D(2, (1,1), activation='sigmoid')(c9)

model = keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())
