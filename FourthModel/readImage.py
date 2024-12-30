import cv2
import matplotlib.pyplot as plt
import numpy as np
from easyocr import easyocr

def digit_found(text):
    """
        Check if the text read contains only one or two digits.

        Parameters:
            text (list): A list containing the text read from the image.

        Returns:
            bool: True if the text contains only one or two digits, False otherwise.
        """
    #If nothing read
    if text:
        txtRes = text[0][1]
        #More than 2 digits
        if len(txtRes) > 2 or len(txtRes) == 1:
            return False

        #check if the string read is composed of digits
        for char in txtRes:
            # Check if the character is a digit
            if not char.isdigit():
                return False
        return True

    else:
        return False




def readImage(OriginalAdaptTresh,ProcessedImage):
    """
        Read the image and detect digits.

        Parameters:
            OriginalAdaptTresh (numpy.ndarray): The original adaptive thresholded image.
            ProcessedImage (numpy.ndarray): The processed image ready to be read

        Returns:
            tuple: A tuple containing the text read from the image, a boolean indicating whether digits are detected, and the processed image.
        """
    digit_detected = False
    newImage=ProcessedImage
    #READING
    reader = easyocr.Reader(['en'], gpu=True)
    # Read text from the image
    text_ = reader.readtext(newImage)


    for i in range (0,11):
        # Flag to track if any digits are detected
        if not digit_found(text_):

            print("NO DIGIT DETECTED")

            #Use the processed image for further processing by using different parameters
            kernelSize=22-2*i
            kernel = np.ones((kernelSize, kernelSize), np.uint8)
            cleaning2 = cv2.morphologyEx(OriginalAdaptTresh, cv2.MORPH_CLOSE, kernel, iterations=1)
            """plt.imshow(cv2.cvtColor(cleaning2, cv2.COLOR_BGR2RGB))
            plt.title("new cleaning2 kernel size : "+str(kernelSize))
            plt.axis("off")
            plt.show()"""

            #Median blur
            median_blurred2 = cv2.medianBlur(cleaning2, 21)
            """plt.imshow(cv2.cvtColor(median_blurred2, cv2.COLOR_BGR2RGB))
            plt.title("new median_blurred2")
            plt.axis("off")
            plt.show()"""

            #Morphological rectification
            ksize=5
            kernel = np.ones((ksize, ksize), np.uint8)
            rect = cv2.morphologyEx(median_blurred2, cv2.MORPH_RECT, kernel, iterations=1)
            """plt.imshow(cv2.cvtColor(rect, cv2.COLOR_BGR2RGB))
            plt.title("new rect kernel size : "+str(ksize))
            plt.axis("off")
            plt.show()"""

            newImage=rect
            #Read the new image

            text_ = reader.readtext(newImage)
        else:
            digit_detected = True
            print(text_)
            return text_, digit_detected, newImage

    return text_,digit_detected,newImage