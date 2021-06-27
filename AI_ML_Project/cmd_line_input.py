import csv
import string

import filetype
from numpy import genfromtxt
from PIL import Image

from AI_ML_Project import accuracy_checker
from os import path
import os.path


class AcceptInput:
    # Checks if the file is a csv file
    def is_csv(infile):
        # Code is cited from stackoverflow
        try:
            with open(infile, newline='') as csvfile:
                start = csvfile.read(4096)

                if not all([c in string.printable or c.isprintable() for c in start]):
                    return False
                dialect = csv.Sniffer().sniff(start)
                return True
        except csv.Error:
            # Could not get a csv dialect -> probably not a csv.
            return False

    # Checks if the input is a string
    def is_string(inp):
        res = isinstance(inp, str)
        if str(res):
            return True
        else:
            return False

    # Checks if the input is an image
    def is_image(filename):
        if filetype.is_image(filename):
            return True
        else:
            return False

    # Converts a csv file to an image
    def convert_csv_to_img(file):
        my_data = genfromtxt(file, delimiter=',')
        im = Image.fromarray(my_data)
        return im
    # Take care of case where CSV has multiple images

    # Checks if the file path is valid
    def is_valid_file(filename):
        if str(path.isfile(filename)):
            return True
        else:
            return False

    def is_valid_folder(file_path):
        if os.path.isdir(file_path):
            return True
        else:
            return False


    def cmd_input(self):
        check_true = False
        user_inp = input("Enter\n")
        space = ' '
        if space in user_inp:
            inp, inp1 = user_inp.split()  # splitting user input
            if self.is_valid_file(inp) or self.is_valid_file(inp1):

                # How to check if the input is a single file or multiple files?
                # 1st case: inp is a the type of image and inp1 is an image
                if self.is_string(inp) and self.is_image(inp1):
                    # Call training class on image
                    #accuracy_checker.accuracy(inp1, inp)

                # 2nd case: inp is a the type of image and inp1 is a csv file
                elif self.is_string(inp) and self.is_csv(inp1):
                    image = self.convert_csv_to_img(inp1)
                    # Not needed
                    #accuracy_checker.accuracy(image, inp)

                # 3rd case: inp is type of image and inp1 is a folder of images
                elif self.is_string(inp) and self.is_valid_folder(inp1):
                    #Now check if folder contains images
                    if not any(fname.endsWith('.jpg') or fname.endswith('.jpeg')
                               for fname in os.listdir(inp1)):
                        check_true = False
                    else:
                        check_true = True

                    if (check_true == True):
                        for img in os.scandir(inp1):
                            if (img.path.endsWith('.jpg') or img.path.endsWith('.jpeg')):
                                # Call training algorithm to train based on this image


        # if no second input is provided
        else:
            if self.is_csv(user_inp):
                image = self.convert_csv_to_img(user_inp)
                accuracy_checker.accuracy(image, user_inp)
            else:
                print("Error: invalid input")

    # Right now the csv file one image
    # Make it work for multiple images
    # Write a csv validator which validates the way we receive data
