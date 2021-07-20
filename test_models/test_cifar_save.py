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
from tensorflow.keras.models import load_model
model = load_model('model_cifar_try1_e5.h5')
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
