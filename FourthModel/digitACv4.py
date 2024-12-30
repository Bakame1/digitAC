import cv2
import matplotlib.pyplot as plt
import numpy as np
from readImage import readImage
from crop import cropImage
import imageProcessing as ip

#Load image
image = cv2.imread('../../../Photos/Aircond/AC (10).jpg')

#Cropping the image
image_cropTab = cropImage(image)
image_cropped, cropSucceed = image_cropTab[0], image_cropTab[1]

txtRes = "empty"
################################## READ THE IMAGE WITH EASYOCR ###################################
if cropSucceed:
    #Image processing after cropping for reading
    image_processed, median_blurred = ip.preprocessImage(image_cropped)

    #READING
    text_, digit_detected, image_processed = readImage(median_blurred, image_processed)

    if digit_detected:
        txtRes = text_[0][1]
        ###############################Draw bounding boxes############################
        bbox = text_[0][0]  # Get the bounding box of the detected text area
        cv2.rectangle(image_processed, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 0), 5)
        cv2.putText(image_processed, txtRes, tuple(map(int, bbox[0])), cv2.FONT_HERSHEY_COMPLEX, 4, (255, 0, 0), 4)

        #Display the image with detected digits highlighted
        plt.imshow(cv2.cvtColor(image_processed, cv2.COLOR_BGR2RGB))
        plt.title("Digits found")
        plt.axis("off")
        plt.show()

    else:
        print("No digit detected")

else :
    print("Cant crop")
