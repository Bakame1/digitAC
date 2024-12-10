import cv2
import numpy as np
from easyocr import easyocr

def model3_f(image_path):
    #Load image
    image = cv2.imread(image_path)

    #Pre-process the image
    image_cropTab=cropImage(image)
    image_cropped,cropSucceed= image_cropTab[0],image_cropTab[1]

    # Apply a bilateral blur to the cropped image to the details
    blurred_image = cv2.bilateralFilter(image_cropped, 50, 75, 75)

    # Use the blurred image for further processing
    image_processed = blurred_image

    txtRes="empty"
    ################################## READ THE IMAGE WITH EASYOCR ###################################
    if cropSucceed:
        #READING
        text_,digit_detected,image_processed=readImage(image_cropped,image_processed)
    
        if digit_detected and text_!=[]:
            txtRes=text_[0][1]
        
        if digit_detected:
            #If we have more than 3 detected digits
            if len(txtRes) > 3:
                #Remove the last digit if it’s a "0" usually it is a the Celsius letter
                if txtRes[-1]== '0':
                    txtRes=txtRes[0]+txtRes[1]  # Remove the last detected "0"
    
                    #Case of S instead of 5
                    if txtRes[1] == "S":
                        txtRes = txtRes[0]+"5"
    
    
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
    
                    #Display txtRes within the adjusted bounding box (only the first two digits)
                    cv2.putText(image_cropped, txtRes[:2], tuple(map(int, bbox[0])), cv2.FONT_HERSHEY_COMPLEX, 4, (255, 0, 0), 4)
    
    
            else:
                if len(txtRes)==2:
                    # Case of S instead of 5
                    if txtRes[1] == "S":
                        txtRes = txtRes[0] + "5"
    
                ###############################Draw bounding boxes############################
                bbox = text_[0][0]  # Get the bounding box of the detected text area
                cv2.rectangle(image_cropped, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 0), 5)
                cv2.putText(image_cropped, txtRes, tuple(map(int, bbox[0])), cv2.FONT_HERSHEY_COMPLEX, 4, (255, 0, 0), 4)
    
        #Return the string result
        return txtRes
    else:
        # Return the string result
        return txtRes


###################################### READ IMAGE#################################


def readImage(OriginalImage,ProcessedImage):
    reader = easyocr.Reader(['en'], gpu=True)
    text_ = reader.readtext(ProcessedImage)
    newImage=ProcessedImage
    digit_detected = False
    for i in range (2):
        # Flag to track if any digits are detected
        if len(text_) == 0:
            # Apply less blurr
            newImage = cv2.bilateralFilter(OriginalImage, 25, 75, 75)

            #Read the new image
            text_ = reader.readtext(newImage)
        else:
            digit_detected = True
            return text_, digit_detected, newImage

    return text_,digit_detected,newImage



###################################### CROPPING IMAGE #################################

def apply_gamma_correction(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def cropImage(im):
    image = im
    if image is None:
        print("Error: Image not found at the specified path.")
        return None, False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_gamma = apply_gamma_correction(gray, gamma=2.2)


    block_sizes = [21, 51, 81]
    C = 2
    largest_cropped_image = None
    largest_width = 0

    for block_size in block_sizes:
        thresh = cv2.adaptiveThreshold(gray_gamma, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, block_size, C)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cropped_image = image[y:y + h, x:x + w]

            if w > largest_width:
                largest_width = w
                largest_cropped_image = cropped_image

    if largest_cropped_image is not None:
        return largest_cropped_image, True
    else:
        print("NO SCREEN DETECTED")
        return image, False