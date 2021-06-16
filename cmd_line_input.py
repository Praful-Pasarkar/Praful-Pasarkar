
class cmd_input:
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
        print(inp)
        print(inp1)
        if word in inp1:
            x = 2  # if only 1 img provided as input
        elif word_csv in inp1:
            x = 3  # if csv is provided as 2nd input
    else:
        x = 0  # if no second input is provided
        print("hi")
        if word_csv in user_inp:
            x = 0  # only csv input
        else:
            x = -1  # invalid input as no training images provided
    #if(len(inp1) == 0):