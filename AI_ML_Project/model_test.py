import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib

class model_test:

  batch_size = 32
  img_height = 180
  img_width = 180

  # Images that will be used for training
  def train_ds(img_height, img_width, batch_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
      'C:\\keys\\flower_photos',
      validation_split=0.2,
      subset="training",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)
      return train_ds

  # Images that will be used for validation
  def val_ds(img_height, img_width, batch_size):
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'C:\\keys\\flower_photos',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
    return val_ds

  # Different types of flowers
  #class_names = train_ds.class_names
  #print(class_names)

  # Scaling down the pixel values
  def normalization(train_ds):
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]

    # The pixels values are now in [0,1].
    print(np.min(first_image), np.max(first_image))

  #AUTOTUNE = tf.data.AUTOTUNE

  #train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
  #val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

  def create_model(train_ds):

    num_classes = 5

    # Doing conv2d 3 times to get a 3d shape
    model = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
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
    return model

  def model_compile(model):
    model.compile(
      optimizer='adam',
      loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])

  def model_fit(model, train_ds, val_ds):
    model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=3
    )

