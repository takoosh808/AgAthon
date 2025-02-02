import keras
from keras import utils

image_directory = "C:/Users/takoo/OneDrive/Desktop/Test/"
dataset = keras.preprocessing.image_dataset_from_directory(image_directory)
print(dataset.__len__)