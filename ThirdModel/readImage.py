import cv2
import matplotlib.pyplot as plt
from easyocr import easyocr

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
    for i in range (2):
        #Flag to track if any digits are detected
        if len(text_) == 0:
            digit_detected = False
            print("NO DIGIT DETECTED")
            print("Let's try with less blurr")
            #Apply less blurr
            newImage = cv2.bilateralFilter(OriginalImage, 25, 75, 75)
            #Display new image
            plt.imshow(newImage)
            plt.title("Image with less blurr")
            plt.axis("off")
            plt.show()
            #Read the new image
            text_ = reader.readtext(newImage)
        else:
            digit_detected = True
            return text_, digit_detected, newImage

    return text_,digit_detected,newImage