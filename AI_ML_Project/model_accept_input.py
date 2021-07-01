import os

class AcceptFolder:
    def cmd_input(self):
        user_inp = input("Enter\n")
        isDirectory = os.path.isdir(user_inp)
        folders = 0

        if isDirectory:
            for _, filenames, dirnames in os.walk(user_inp):
                folders += len(dirnames)
            return folders

        else:
            return 'Not a valid directory'

