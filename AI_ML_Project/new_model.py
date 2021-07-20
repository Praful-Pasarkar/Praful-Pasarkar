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
from tensorflow.keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
#from skimage.transform import resize

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
  img_height = 180
  img_width = 180


  # Images that will be used for training
  train_ds = tf.keras.preprocessing.image_dataset_from_directory(
      'C:\\Keys\\idenprof\\test',
      validation_split=0.001,
      subset="training",
      shuffle=True,
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)

  # Images that will be used for validation
  val_ds = tf.keras.preprocessing.image_dataset_from_directory(
      'C:\\Keys\\idenprof\\train',
      validation_split=0.999,
      subset="validation",
      shuffle=True,
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)



  # Different types of flowers
  # class_names = train_ds.class_names
  # print(class_names)

  # Scaling down the pixel values
  def normalization(train_ds):
      normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
      normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
      image_batch, labels_batch = next(iter(normalized_ds))
      first_image = image_batch[0]

      # The pixels values are now in [0,1].
      print(np.min(first_image), np.max(first_image))
      print('Normalization function')

  def create_model(train_ds, self=None):
      num_classes = 10
      #num_classes = model_accept_input.AcceptFolder.cmd_input(self)
      #print(num_classes)

      # Doing conv2d 3 times to get a 3d shape
      # tf.keras.layers.Conv2D(32, 3, activation='relu'),
      # tf.keras.layers.MaxPooling2D(),
      # # tf.keras.layers.Flatten(),
      model = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255),
          tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(180,180,3)),
          tf.keras.layers.MaxPooling2D(),
          tf.keras.layers.Conv2D(32, 3, activation='relu'),
          tf.keras.layers.MaxPooling2D(),
          tf.keras.layers.Conv2D(32, 3, activation='relu'),
          tf.keras.layers.MaxPooling2D(),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(128, activation='relu'),
          tf.keras.layers.Dense(num_classes)
      ])
      #model.summary()
      print('Creating model')
      return model

  def model_compile(model):
      model.compile(
          optimizer='rmsprop',   #rmsprop
          loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=['accuracy'])

      print('Compiling model')
      return model

  def model_fit(model, train_ds, val_ds):
    model.fit(train_ds,validation_data=val_ds,epochs=5)

    model.summary()

    print('Executing model')

  def is_image(filename):
      if filetype.is_image(filename):
          return True
      else:
          return False

  def classify_image(train_ds, model, dirlist):
      inp1 = 'y'
      while inp1 == 'y' or inp1 == 'Y':
          inp = input("Enter img path\n")
          if is_image(inp):
              # img = Image.open(inp)
              # img = img.resize((180, 180))
              # #img = resize(img, (180,180,3))
              # predictions = model.predict(np.array(img))
              img_path = inp
              img = image.load_img(img_path, target_size=(180, 180))
              img_array = image.img_to_array(img)
              img_batch = np.expand_dims(img_array, axis=0)
              img_preprocessed = preprocess_input(img_batch)
              predictions = model.predict(np.array(img_preprocessed))
              list_index = [0,1,2,3,4,5,6,7,8,9]
              x = predictions
              print(x,'\n')
              # for i in range(10):
              #     for j in range(10):
              #         if x[i] > x[j]:
              #             temp = list_index[i]
              #             list_index[i] = list_index[j]
              #             list_index[j] = temp
              #
              # for i in range(5):
              #     print(dirlist[list_index[i]], ':', predictions[0][list_index[i]] * 100, '%')
              inp1 = input("Do you want to confinue?\n")

  def image_size(self):
      inp = input("Enter\n")
      image = PIL.Image.open(inp)
      width, height = image.size
      print(width, height)

  # Prints the names of all the folders in a directory
  def folder_names(self):
      root = input("Enter folder path\n")
      dirlist = [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))]
      print(dirlist)
      return dirlist

  #normalization(train_ds)
  model = create_model(train_ds)
  model1 = model_compile(model)
  model_fit(model1, train_ds, val_ds)
  dirlist = folder_names(self=None)
  classify_image(train_ds, model, dirlist)
  #image_size(self=None)
