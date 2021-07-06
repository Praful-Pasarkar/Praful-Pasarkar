from PIL import Image
import filetype

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
        if self.is_image(inp):
            # Send it to another class that will identify it

    # The trained model doesn't return image type.

