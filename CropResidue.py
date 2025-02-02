import numpy as np
np.random.seed(1000)
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import cv2
import keras
from PIL import Image


image_directory = "C:/Users/takoo/OneDrive/Desktop/Images/Training/"
dataset = keras.preprocessing.image_dataset_from_directory(image_directory, labels=None, image_size=(128,128), batch_size=32)



IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3

input = keras.layers.Input(shape=(128,128,3))

s = keras.layers.Lambda(lambda x: x / 255)(input)

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

model = keras.Model(inputs=[s], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

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