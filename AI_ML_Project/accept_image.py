import PIL
from PIL import Image
import filetype
import numpy as np
from keras.preprocessing import image
import model_test

class AcceptImage:

    # Checks if the input is an image
    def is_image(filename):
        if filetype.is_image(filename):
            return True
        else:
            return False

    # Sends the image to another class to identify it
    def user_input(self): # Supply array of images to model.py
        inp = input("Enter\n")
        image = PIL.Image.open(inp)
        width, height = image.size

        print(width, height)

