import csv
import string

import filetype
from numpy import genfromtxt
from PIL import Image

import accuracy_checker
import model

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
    my_data = genfromtxt('my_file.csv', delimiter=',')
    im = Image.fromarray(my_data)
    return im

class CmdInput:
    user_inp = input("Enter\n")
    space = ' '
    if space in user_inp:
        inp, inp1 = user_inp.split()  # splitting user input

        # 1st case: inp is a the type of image and inp1 is an image
        if is_string(inp) and is_image(inp1):
            accuracy_checker.accuracy(inp1, inp)

        # 2nd case: inp is a the type of image and inp1 is a csv file
        elif is_string(inp) and is_csv(inp1):
            image = convert_csv_to_img(inp1)
            accuracy_checker.accuracy(image, inp)

    # if no second input is provided
    else:
        if is_csv(user_inp):
            image = convert_csv_to_img(user_inp)
            accuracy_checker.accuracy(image, user_inp)
        else:
            print("Error: invalid input")

# Right now the csv file one image
