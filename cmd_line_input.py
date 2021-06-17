import csv
import string
import model


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


class CmdInput:

    user_inp = input("Enter\n")
    space = ' '
    space_check = -1
    word = '.img'
    word_csv = '.csv'
    x = 1  # default option assuming input is a folder of images
    if space in user_inp:
        space_check = 0
    if space_check == 0:
        inp, inp1 = user_inp.split()  # splitting user input
        if is_csv(inp):
            data = list(csv.reader(open(inp)))
            model.identify_image(data)  #sends a 2d list containing all the images to the model class
        elif is_csv(inp1):
            data = list(csv.reader(open(inp)))
            model.identify_image(data)  #sends a 2d list containing all the images to the model class
    else:
        x = 0  # if no second input is provided
        print("Error: invalid input")
        if word_csv in user_inp:
            x = 0  # only csv input
        else:
            x = -1  # invalid input as no training images provided
    # if(len(inp1) == 0):
