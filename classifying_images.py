# Basic class to read the arguments 
# three  types of arguments:
# 1. python3 train_test <train object name> abc.jpg
# 2. python3 train_test <train object name> abc.jpg, pqr.jpg, poir.jpg  (look at the comma)
# 3. python3 train_test <train object name> abc.jpg pqr.jpr poir.jpg    (look at the space can be 1+ spaces)
# 4. python3 train_test abc.csv (Train object name is inside the csv file)
# 5. python3 train_test <train object name> abc.csv
# what happens when the <train object name> is not given?


# File for reference: Input will be taken from cmd_line_input.py

import csv
import accuracy_checker


class classifying_images:

    # input is: image type and image
    def input_check(img_type, img):
        print(img_type)
        accuracy_checker.accuracy(img, img_type)
        return img_type

    # input is: csv file
    def input_csv(img_csv):
        with open(img_csv, newline='') as f:
            reader = csv.reader(f)
            image_type = next(reader)  # gets the first line

        accuracy_checker.accuracy('image', image_type)  # is each line a new image in the csv file?
        return image_type

    # input is: image type and csv file
    def input_csv_2(img_type, img_csv):
        print(img_type)
        accuracy_checker.accuracy(img_type)
        return img_type
