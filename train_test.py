# Basic class to read the arguments 
# three  types of arguments:
# 1. python3 train_test <train object name> abc.jpg
# 2. python3 train_test <train object name> abc.jpg, pqr.jpg, poir.jpg  (look at the comma)
# 3. python3 train_test <train object name> abc.jpg pqr.jpr poir.jpg    (look at the space can be 1+ spaces)
# 4. python3 train_test abc.csv (Train object name is inside the csv file)
# 5. python3 train_test <train object name> abc.csv
# what happens when the <train object name> is not given?

class basic :
    def input_check(img_type,img):
        print(img_type)
        return img_type

    def input_csv_2(img_type, img_csv):
        print(img_type)
        return img_type

    def input_csv(img_csv):
        print("Image CSV folder")
        return "Image CSV folder"

    def check(a):
        return a
