import cv2
import numpy as np
from easyocr import easyocr

def model2_f(image_path):
    """
        Process the image and detect the temperature.
        Here we divided the cropping and the reading part

        Parameters:
            image_path (str): The path to the image file.

        Returns:
            str: The detected temperature.
        """
    #Load image
    image = cv2.imread(image_path)

    #Pre-process the image
    image_cropped= cropImage(image)

    # Apply a bilateral blur to the cropped image to the details
    blurred_image = cv2.bilateralFilter(image_cropped, 50, 75, 75)

    image_processed = blurred_image


    ################################## READ THE IMAGE WITH EASYOCR ###################################
    text_,digit_detected,image_processed=readImage(image_cropped,image_processed)

    if digit_detected and text_!=[]:
        txtCopy=text_[0][1]
    else:
        txtCopy="empty"

    if digit_detected :
        #If we have more than 3 detected digits
        if len(txtCopy) > 3:
            #Remove the last digit if it’s a "0" usually it is the Celsius letter
            if txtCopy[-1]== '0':
                txtCopy=txtCopy[0]+txtCopy[1]  #Remove the last detected "0"

                #Case of S instead of 5
                if txtCopy[1] == "S":
                    txtCopy = txtCopy[0]+"5"


            ####################Draw bounding boxes#############################
            if len(text_) >= 1:
                bbox = text_[0][0]  #Get the bounding box of the detected text area

                #Calculate the approximate width of each digit and the width of the first two digits
                digit_widths = [(bbox[1][0] - bbox[0][0]) / len(text_[0][1])]  # Average width per character
                first_two_digits_width = int(digit_widths[0] * 2)  # Width for first two characters only

                #The bottom right corner of the rectangle we want to draw
                adjusted_bottom_right = (int(bbox[0][0] + first_two_digits_width), int(bbox[2][1]))

                #Draw the rectangle with adjusted width to cover only the first two digits
                cv2.rectangle(image_cropped, tuple(map(int, bbox[0])), adjusted_bottom_right, (0, 255, 0), 2)

                #Display txtCopy within the adjusted bounding box (only the first two digits)
                cv2.putText(image_cropped, txtCopy[:2], tuple(map(int, bbox[0])), cv2.FONT_HERSHEY_COMPLEX, 4, (255, 0, 0), 4)


        else:
            if len(txtCopy)==2:
                #Case of S instead of 5
                if txtCopy[1] == "S":
                    txtCopy = txtCopy[0] + "5"

            ###############################Draw bounding boxes############################
            bbox = text_[0][0]  # Get the bounding box of the detected text area
            cv2.rectangle(image_cropped, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 0), 5)
            cv2.putText(image_cropped, txtCopy, tuple(map(int, bbox[0])), cv2.FONT_HERSHEY_COMPLEX, 4, (255, 0, 0), 4)

    #Return the string result
    return(txtCopy)

###################################### READ IMAGE#################################


def readImage(OriginalImage,ProcessedImage):
    """
        Read the image and detect digits.

        Parameters:
            OriginalImage (numpy.ndarray): The original image in BGR format.
            ProcessedImage (numpy.ndarray): The processed image.

        Returns:
            tuple: A tuple containing the text read from the image, a boolean indicating whether digits are detected, and the processed image.
        """
    reader = easyocr.Reader(['en'], gpu=True)
    text_ = reader.readtext(ProcessedImage)
    newImage=ProcessedImage
    digit_detected = False
    for i in range (2):
        #Flag to track if any digits are detected
        if len(text_) == 0:
            #Apply less blurr
            newImage = cv2.bilateralFilter(OriginalImage, 25, 75, 75)

            #Read the new image
            text_ = reader.readtext(newImage)
        else:
            digit_detected = True
            return text_, digit_detected, newImage

    return text_,digit_detected,newImage

###################################### CROPPING IMAGE #################################




def cropImage(im):
    """
        Crop the image to extract the region of interest.

        Parameters:
            im (numpy.ndarray): The input image in BGR format.

        Returns:
            numpy.ndarray: The cropped image.
        """
    image = im
    cropped_image=im

    #HSV conversion
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #Define threshold colors (gray and green)
    lower_bound = np.array([30, 10, 10])   # Teinte, Saturation, Valeur minimales
    upper_bound = np.array([90, 50, 200])  # Teinte, Saturation, Valeur maximales

    #Create a mask based on thresholds
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    #contours based on the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Find the biggest contour
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        #Cropping
        cropped_image = image[y:y+h, x:x+w]

    else:
        print("NO SCREEN DETECTED")

    return cropped_image

