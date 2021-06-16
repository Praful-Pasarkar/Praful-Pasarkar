import model

class accuracy_checker:
    #checks the accuracy with which a the model identifies the image

    def accuracy(image, actual_image_type):
        count = 0
        model_result = model.identify_image(image)
        if model_result == actual_image_type:
            count += 1
            print("Correctly identified")


    #Need another function for the csv file accuracy check
    #Should while loop for the csv file/image bank in order to get percentage accuracy
    #For machine learning will we use naive bayes?
