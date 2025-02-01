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

image_directory = "C:/Users/takoo/OneDrive/Desktop/Images/Training/"
trainset = []
label = []

cropResidue_images = os.listdir(image_directory + "CropResidue/")
for i, image_name in enumerate(cropResidue_images):
    if (image_name.split('.')[1] == 'jpg' or image_name.split('.')[1] == 'tif'):
        image = cv2.imread(image_directory + "CropResidue/" + image_name)
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
inp = keras.layers.Input(shape=INPUT_SHAPE)

conv1 = keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inp)

pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
norm1 = keras.layers.BatchNormalization(axis=-1)(pool1)
drop1 = keras.layers.Dropout(0.2)(norm1)

conv2 = keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(drop1)

pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
norm2 = keras.layers.BatchNormalization(axis=-1)(pool2)
drop2 = keras.layers.Dropout(0.2)(norm2)

flat = keras.layers.Flatten()(drop2)

hidden1 = keras.layers.Dense(512, activation='relu')(flat)
norm3 = keras.layers.BatchNormalization(axis=-1)(hidden1)
drop3 = keras.layers.Dropout(0.2)(norm3)

hidden2 = keras.layers.Dense(256, activation='relu')(drop3)
norm4 = keras.layers.BatchNormalization(axis=-1)(hidden2)
drop4 = keras.layers.Dropout(0.2)(norm4)

out = keras.layers.Dense(2, activation='sigmoid')(drop4)

model = keras.Model(inputs=inp, outputs=out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.utils import to_categorical
X_train, X_test, y_train, y_test = train_test_split(trainset, to_categorical(np.array(label)), test_size=0.2, random_state=42)

history = model.fit(np.array(X_train), y_train, batch_size = 64, verbose = 1, epochs = 25, validation_split = 0.1, shuffle = False)

print("Test_Accuracy: {:.2f}%".format(model.evaluate(np.array(X_test), np.array(y_test))[1]*100))
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
