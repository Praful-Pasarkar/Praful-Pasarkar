import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.datasets import cifar10
from tensorflow.keras import layers
from keras.utils import to_categorical
import numpy as np

class new_model:
    train_dataset = ImageDataGenerator(rescale=1. / 255)
    validation_dataset = ImageDataGenerator(rescale=1. / 255)

