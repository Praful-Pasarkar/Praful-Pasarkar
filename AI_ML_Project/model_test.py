import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import os
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

# C:\keys\idenprof\train\chef
# C:\\keys\\idenprof\\train

from filetype import is_image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import model_accept_input
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
      'C:\\keys\\idenprof\\train',
      validation_split=0.2,
      subset="training",
      shuffle=True,
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)

  # Images that will be used for validation
  val_ds = tf.keras.preprocessing.image_dataset_from_directory(
      'C:\\keys\\idenprof\\test',
      validation_split=0.2,
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
      #num_classes = 5
      num_classes = model_accept_input.AcceptFolder.cmd_input(self)
      print(num_classes)

      # Doing conv2d 3 times to get a 3d shape
      # tf.keras.layers.Conv2D(32, 3, activation='relu'),
      # tf.keras.layers.MaxPooling2D(),
      # tf.keras.layers.Flatten(),
      model = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255),
          tf.keras.layers.Conv2D(32, 3, activation='relu'),
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
          optimizer='adam',   #rmsprop
          loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=['accuracy'])

      print('Compiling model')
      return model

  def model_fit(model, train_ds, val_ds):
    model.fit(train_ds,validation_data=val_ds,epochs=10)

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
              img_path = inp
              img = image.load_img(img_path, target_size=(180, 180))
              img_array = image.img_to_array(img)
              img_batch = np.expand_dims(img_array, axis=0)
              img_preprocessed = preprocess_input(img_batch)
              #model = tf.keras.applications.resnet50.ResNet50()
              prediction = np.argmax(model.predict(img_preprocessed), axis=-1)  #model.predict_classes(img_preprocessed)
              predictions = prediction.reshape(1, -1)[0]
              classname = prediction[0]
              image_type = dirlist[classname - 1]
              print("Image type: ", image_type)
              #print(np.argmin(prediction[0]))
              #predictions = model.predict(img)
              #print(dirlist[np.argmax (predictions[0])])

              # Resizing the image and sending it to model
              # img = Image.open(inp)
              # img = img.resize((180, 180))
              # img = np.array(img)
              # #img = img / 255.0
              # img = img.reshape(1, 180, 180, 3)
              # img_class = model.predict(img) #np.argmax(model.predict(img), axis=-1)  # Predicts the image type
              # prediction, percentage = model.classifyImage("C:\\Users\\manas\\Downloads\\daisy1.jpg")
              # for index in range(len(prediction)):
              #    print(prediction[index], " : ", percentage[index])
              # classname = img_class[0]   # Returns the index of the folder
              # image_type = dirlist[classname - 1]  # Gets the name of the folder
              #print("Class: ", image_type)
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
