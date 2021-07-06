import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import model_accept_input


class model_test:
  os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
  batch_size = 32
  img_height = 180
  img_width = 180

  # Images that will be used for training
  train_ds = tf.keras.preprocessing.image_dataset_from_directory(
      'C:\\keys\\flower_photos',
      validation_split=0.1,
      subset="training",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)

  # Images that will be used for validation
  val_ds = tf.keras.preprocessing.image_dataset_from_directory(
      'C:\\keys\\flower_photos',
      validation_split=0.5,
      subset="validation",
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
      print('Creating model')
      return model

  def model_compile(model):
      model.compile(
          optimizer='adam',
          loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=['accuracy'])

      print('Compiling model')

  def model_fit(model, train_ds, val_ds):
    model.fit(train_ds,validation_data=val_ds,epochs=5)

    print('Executing model')

  normalization(train_ds)
  model = create_model(train_ds)
  model_compile(model)
  model_fit(model, train_ds, val_ds)