#import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

#from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf

import cv2
import os

import numpy as np

class new_model:

    #labels = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    labels = ['daisy', 'dandelion']
    img_size = 224
    def get_data(data_dir, labels, img_size):
        data = []
        for label in labels:
            path = os.path.join(data_dir, label)
            class_num = labels.index(label)
            for img in os.listdir(path):
                try:
                    img_arr = cv2.imread(os.path.join(path, img))#[...,::-1] #convert BGR to RGB format
                    resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                    data.append([resized_arr, class_num])
                except Exception as e:
                    print(e)
        return np.array(data, dtype=object)

    # def show_data(train):
    #     l = []
    #     for i in train:
    #         if (i[1] == 0):
    #             l.append("rugby")
    #         else:
    #             l.append("soccer")
    #     sns.set_style('darkgrid')
    #     sns.countplot(l)

    def model_compile(model):
        model.compile(
            optimizer='adam',
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

        print('Compiling model')

        return model

    def model_fit(model, x_train, y_train, x_val, y_val):
        model.fit(x_train,y_train,epochs = 5 , validation_data = (x_val, y_val))

        print('Executing model')

        return model

    train = get_data('C:\\keys\\temp_train', labels, img_size)
    val = get_data('C:\\keys\\temp_val', labels, img_size)

    x_train = []
    y_train = []
    x_val = []
    y_val = []

    for feature, label in train:
      x_train.append(feature)
      y_train.append(label)

    for feature, label in val:
      x_val.append(feature)
      y_val.append(label)

    # Normalize the data
    x_train = np.array(x_train) / 255
    x_val = np.array(x_val) / 255

    x_train.reshape(-1, img_size, img_size, 1)
    y_train = np.array(y_train)

    x_val.reshape(-1, img_size, img_size, 1)
    y_val = np.array(y_val)

    datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range = 30,
            zoom_range = 0.2,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip = True,
            vertical_flip=False)

    datagen.fit(x_train)

    model = Sequential()
    model.add(Conv2D(32, 3, padding="same", activation="relu", input_shape=(224, 224, 3)))
    model.add(MaxPool2D())

    model.add(Conv2D(32, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())

    model.add(Conv2D(64, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(2, activation="softmax"))

    model.summary()

    model1 = model_compile(model)
    model_fit(model1, x_train, y_train, x_val, y_val)




