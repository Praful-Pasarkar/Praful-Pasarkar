# This class will be use machine learning to identify the type of image

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.datasets import cifar10
from tensorflow.keras import layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt


class Model:

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck ']
    y_train_one_hot = to_categorical(y_train)
    y_test_one_hot = to_categorical(y_test)

    # Normalizing pixels to have a value between 0 and 1
    x_train = x_train / 255
    x_test = x_test / 255

    # Creating the layers of model
    model = Sequential()

    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(32, 32, 3)))  # First layer
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Pooling layer

    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(10, activation='relu'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Training the model
    histogram = model.fit(x_train, y_train_one_hot, batch_size=256, epochs=10, validation_split=0.2)

    model.evaluate(x_test, y_test_one_hot)[1]

    # Plotting the model's accuracy
    plt.plot(histogram.history['accuracy'])
    plt.plot(histogram.history['val_accuracy'])
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()
