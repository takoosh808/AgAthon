import numpy as np
np.random.seed(1000)
from sklearn.model_selection import train_test_split
from keras import applications
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import cv2
import keras
from PIL import Image
import os

gpu_index = os.getenv("CUDA_VISIBLE_DEVICES")
print(f"Using GPU: {gpu_index}")
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index

image_directory = "/scratch/project/hackathon/data/CropResiduePredictionChallenge/images_512/original/Limbaugh1-1m20220328"


IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3

input = keras.layers.Input(shape=(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS))

#using resnet50 (pretrained processor) as backbone
base_model = keras.applications.resnet50(input_tensor=input, include_top=False, weights='imagenet')
base_model.trainable=False #freeze backbone during initial training

s = keras.layers.Lambda(lambda x: x / 255)(input)

c1 = base_model.get_layer("conv1_relu").output #64 filters
c2 = base_model.get_layer("conv2_block3_out").output #256 filters
c3 = base_model.get_layer("conv3_block4_out").output #512 filters
c4 = base_model.get_layer("conv4_block6_out").output #1024 filters

#Bottleneck (lowest level of u-net)
c5 = keras.layers.Conv2D(2048, (3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(c4)
c5 = keras.layers.Dropout(0.3)(c5)
c5 = keras.layers.Conv2D(2048, (3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(c5)

u6 = keras.layers.Conv2DTranspose(1024, (2,2), strides = (2,2), padding = 'same')(c5)
u6 = keras.layers.Concatenate()([u6,c4])
c6 = keras.layers.Conv2D(1024, (3,3), activation ='relu', kernel_initializer='he_normal', padding = 'same')(u6)
c6 = keras.layers.Dropout(0.2)(c6)
c6 = keras.layers.Conv2D(1024, (3,3), activation ='relu', kernel_initializer='he_normal', padding = 'same')(c6)

u7 = keras.layers.Conv2DTranspose(512, (2,2), strides = (2,2), padding = 'same')(c6)
u7 = keras.layers.Concatenate()([u7, c3])
c7 = keras.layers.Conv2D(512, (3,3), activation ='relu', kernel_initializer='he_normal', padding = 'same')(u7)
c7 = keras.layers.Dropout(0.2)(c7)
c7 = keras.layers.Conv2D(512, (3,3), activation ='relu', kernel_initializer='he_normal', padding = 'same')(c7)

u8 = keras.layers.Conv2DTranspose(256, (2,2), strides = (2,2), padding = 'same')(c7)
u8 = keras.layers.Concatenate()([u8, c2])
c8 = keras.layers.Conv2D(256, (3,3), activation ='relu', kernel_initializer='he_normal', padding = 'same')(u8)
c8 = keras.layers.Dropout(0.1)(c8)
c8 = keras.layers.Conv2D(256, (3,3), activation ='relu', kernel_initializer='he_normal', padding = 'same')(c8)

u9 = keras.layers.Conv2DTranspose(128, (2,2), strides = (2,2), padding = 'same')(c8)
u9 = keras.layers.Concatenate()([u9, c1])
c9 = keras.layers.Conv2D(128, (3,3), activation ='relu', kernel_initializer='he_normal', padding = 'same')(u9)
c9 = keras.layers.Dropout(0.1)(c9)
c9 = keras.layers.Conv2D(128, (3,3), activation ='relu', kernel_initializer='he_normal', padding = 'same')(c9)

outputs = keras.layers.Conv2D(1, (1,1), activation='sigmoid')(c9)

model = keras.Model(inputs=[input], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())
def load_dataset(img_directory, batch_size=32, img_size=(IMG_HEIGHT,IMG_WIDTH)):
    dataset = keras.preprocessing.image_dataset_from_directory(
        img_directory, 
        labels="inferred", 
        label_mode="binary",
        image_size=img_size,
        batch_size=batch_size
    )
    return dataset

dataset=load_dataset(image_directory)

for elem in dataset:
    x_train = elem
    y_train = elem   
    (x_train, y_train), (x_test,y_test) = train_test_split(x_train,y_train, train_size=0.8, test_size=0.1, random_state=32)
    history = model.fit(x = x_train, y = y_train,verbose = 1, epochs=5,steps_per_epoch=32, validation_split=0.1, shuffle = False)
    print("Test_Accuracy: {:.2f}%".format(model.evaluate(np.array(x_test), np.array(y_test))[1]*100))
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    t = f.suptitle('CNN Performance', fontsize=12)
    f.subplots_adjust(top=0.85, wspace=0.3)
    max_epoch = len(history.history['accuracy'])+1
    epoch_list = list(range(1,max_epoch))
    ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
    ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_xticks(np.arange(1, max_epoch, 5))
    ax1.set_ylabel('Accuracy Value')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Accuracy')
    l1 = ax1.legend(loc="best")
    ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
    ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
    ax2.set_xticks(np.arange(1, max_epoch, 5))
    ax2.set_ylabel('Loss Value')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Loss')
    l2 = ax2.legend(loc="best")


model.save('cropResidue.h5')