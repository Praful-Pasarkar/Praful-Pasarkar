import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
# import tensorflow_datasets as tfds
import pathlib
import os
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
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

# C:\keys\idenprof\train\chef
# C:\\keys\\idenprof\\train

from filetype import is_image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#import model_accept_input
from PIL import Image
import filetype
from keras.preprocessing import image


class model_test:
  os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
  batch_size = 32
  img_height = 224
  img_width = 224


  # Images that will be used for training
  train_ds = tf.keras.preprocessing.image_dataset_from_directory(
      'C:\\Keys\\idenprof\\train',
      validation_split=0.001,
      subset="training",
      shuffle=True,
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)

  # Images that will be used for validation
  val_ds = tf.keras.preprocessing.image_dataset_from_directory(
      'C:\\Keys\\idenprof\\test',
      validation_split=0.999,
      subset="validation",
      shuffle=True,
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)

  num_classes = 10
  # train_ds = train_ds.astype('float32')
  # val_ds = val_ds.astype('float32')
  # train_ds = train_ds / 255.0
  # val_ds = val_ds / 255.0
  # train_ds = np_utils.to_categorical(train_ds)
  # val_ds = np_utils.to_categorical(val_ds)
  #num_classes = model_accept_input.AcceptFolder.cmd_input(self)
  #print(num_classes)
  model = Sequential()
  model.add(Conv2D(224, (3, 3), input_shape=(224, 224, 3),
              padding='same', activation='relu',
             kernel_constraint=maxnorm(3)))
  model.add(Dropout(0.2))
  model.add(Conv2D(224, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes, activation='softmax'))
      #model.summary()
  print('Creating model')

  model.compile(loss='categorical_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])

  print('Compiling model')

  model.fit(train_ds,validation_data=val_ds,epochs=5)

  model.summary()

  print('Executing model')

  def is_image(filename):
      if filetype.is_image(filename):
          return True
      else:
          return False

  def image_size(self):
      inp = input("Enter\n")
      image = PIL.Image.open(inp)
      width, height = image.size
      print(width, height)

  # Prints the names of all the folders in a directory
  def folder_names(root):
      dirlist = [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))]
      print(dirlist)
      return dirlist


  root = input("Enter folder path\n")
  dirlist = folder_names(root)
  results = dirlist
  check = 'y'
  while check == 'y' or check == 'Y':
      inp = input("Enter img path \n")
      im = Image.open(inp)
      im = im.resize((224, 224))
      im = np.expand_dims(im, axis=0)
      im = np.array(im)
      pred = model.predict_classes([im])[0]
      print(pred, ' \n')
      print(results[pred])
      check = input("Enter Y or N \n")

  # #normalization(train_ds)
  # model = create_model(train_ds)
  # model1 = model_compile(model)
  # model_fit(model1, train_ds, val_ds)
  # dirlist = folder_names(self=None)
  # classify_image(train_ds, model, dirlist)
  #image_size(self=None)

