import csv
import string


class cmd_input:

    def accept_input():
        user_inp = input("Enter\n")
        space = ' '
        space_check = -1
        word = '.img'
        word_csv = '.csv'
        x = 1  # default option assuming input is a folder of images
        if space in user_inp:
            space_check = 0
        if (space_check == 0):
            inp,inp1 = user_inp.split() #splitting user input
            #print(inp)
            #print(inp1)


            if is_csv(inp) == True:
                x = 2  # if only 1 img provided as input
            elif is_csv(inp1) == True:
                x = 3  # if csv is provided as 2nd input
        else:
            x = 0  # if no second input is provided
            print("hi")
            if word_csv in user_inp:
                x = 0  # only csv input
            else:
                x = -1  # invalid input as no training images provided
        #if(len(inp1) == 0):

    def is_csv(infile):

        #Code is cited from stackoverflow
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