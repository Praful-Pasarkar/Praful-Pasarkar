import os
import tensorflow as tf
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import image_dataset_loader

class cifarpng:
    (train_X, train_Y), (test_X, test_Y) = image_dataset_loader.load('C:\\Keys\\cifar10', ['train', 'test'])
    train_x = train_X.astype('float32')
    test_X = test_X.astype('float32')

    train_X = train_X / 255.0
    test_X = test_X / 255.0
    train_Y = np_utils.to_categorical(train_Y)
    test_Y = np_utils.to_categorical(test_Y)

    num_classes = test_Y.shape[1]
    print(num_classes)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3),
                     padding='same', activation='relu',
                     kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.summary()
    model.fit(train_X, train_Y,
              validation_data=(test_X, test_Y),
              epochs=10, batch_size=32)
    model.save("model_cifar_final.h5")

    results = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    check = 'y'
    while check == 'y' or check == 'Y':
        inp = input("Enter img path \n")
        im = Image.open(inp)
        im = im.resize((32, 32))
        im = np.expand_dims(im, axis=0)
        im = np.array(im)
        pred = model.predict_classes([im])[0]
        print(pred, ' \n')
        print(results[pred])
        check = input("Enter Y or N \n")