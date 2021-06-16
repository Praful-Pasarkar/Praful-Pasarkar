# Basic class to read the arguments 
# three  types of arguments:
# 1. python3 train_test <train object name> abc.jpg
# 2. python3 train_test <train object name> abc.jpg, pqr.jpg, poir.jpg  (look at the comma)
# 3. python3 train_test <train object name> abc.jpg pqr.jpr poir.jpg    (look at the space can be 1+ spaces)
# 4. python3 train_test abc.csv (Train object name is inside the csv file)
# 5. python3 train_test <train object name> abc.csv
# what happens when the <train object name> is not given?

import csv
import accuracy_checker

class classifying_images:

    #imput is: image type and image
    def input_check(img_type,img):
        print(img_type)
        accuracy_checker.accuracy(img, img_type)
        return img_type

    #imput is: csv file
    def input_csv(img_csv):
        with open(img_csv, newline='') as f:
            reader = csv.reader(f)
            image_type = next(reader)  # gets the first line

        accuracy_checker.accuracy('image', image_type)  #where will the image be in the csv file?
        return image_type

    # imput is: csv file and image type
    def input_csv_2(img_type, img_csv):
        print(img_type)
        accuracy_checker.accuracy(img_type)
        return img_type


