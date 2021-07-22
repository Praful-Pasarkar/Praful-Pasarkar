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

class cifarpng:
    DATA_DIR = "C:\\Keys\\cifar10\\train"
    #DATA_DIR = "C:\\Keys\\idenprof\\train"
    CATERGORIES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    #CATERGORIES = ['chef', 'doctor', 'engineer', 'farmer', 'firefighter', 'judge', 'mechanic', 'pilot', 'police', 'waiter']
    IMAGE_SIZE = 32
    training_data = []
    for categories in CATERGORIES:
        path = os.path.join(DATA_DIR, categories)
        class_num = CATERGORIES.index(categories)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
                training_data.append([new_array, class_num])
            except:
                pass
    data = np.asarray(training_data)
    x_data = []
    y_data = []

    for x in data:
        x_data.append(x[0])
        y_data.append(x[1])
    x_data_np = np.asarray(x_data)
    y_data_np = np.asarray(y_data)

    x_data_np = x_data_np.reshape(-1, 32, 32, 1)
    print(x_data_np.shape)
    print(y_data_np.shape)

    (train_X, train_Y), (test_X, test_Y) = train_test_split(x_data_np, y_data_np, test_size=0.3, random_state=101)
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
              epochs=5, batch_size=32)
    #model.save("model_cifar_try1_e5.h5")

    results = ['aeroplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
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