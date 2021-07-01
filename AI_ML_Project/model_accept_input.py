import os

class AcceptFolder:
    def cmd_input(self):
        user_inp = input("Enter\n")
        isDirectory = os.path.isdir(user_inp)
        folder_count = 0

        if isDirectory:
            for folders in os.listdir(user_inp):
                folder_count += 1
            return folder_count

        else:
            return 'Not a valid directory'

