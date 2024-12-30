import cv2
import matplotlib.pyplot as plt
import numpy as np
from easyocr import easyocr


def model4_f(image_path):
    """
        Process the image and detect the temperature.
        Improved image quality and processing

        Parameters:
            image_path (str): The path to the image file.

        Returns:
            str: The detected temperature.
        """
    #Load image
    image = cv2.imread(image_path)

    #Pre-process the image
    image_cropTab = cropImage(image)
    image_cropped, cropSucceed = image_cropTab[0], image_cropTab[1]



    txtRes = "empty"
    ################################## READ THE IMAGE WITH EASYOCR ###################################
    if cropSucceed:
        #Use the processed image for further processing before reading
        image_processed, median_blurred = preprocessImage(image_cropped)

        #READING
        text_, digit_detected, image_processed = readImage(median_blurred, image_processed)

        if digit_detected:
            txtRes = text_[0][1]
            return txtRes
        else:

            print("No digit detected")
            return "empty"
    else :
        print("Cant crop")
        return "empty"


################################# CROP THE IMAGE #######################
def apply_gamma_correction(image, gamma=1.0):
    """
        Apply gamma correction to the image.

        Parameters:
            image (numpy.ndarray): The input image in BGR format.
            gamma (float): The gamma value. Default is 1.0.

        Returns:
            numpy.ndarray: The gamma-corrected image.
        """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def cropImage(im):
    """
        Crop the image to extract the region of interest.

        Parameters:
            im (numpy.ndarray): The input image in BGR format.

        Returns:
            tuple: A tuple containing the cropped image and a boolean indicating whether the cropping was successful.
        """
    image=im
    #if incorrect path
    if image is None:
        print("Error: Image not found at the specified path.")
        return None, False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_gamma = apply_gamma_correction(gray, gamma=2.2)

    # Apply adaptive thresholding with different block sizes
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
        return image, False

########################################### IMAGE PROCESSING ###########################################################

def preprocessImage(image_cropped):
    """
        Preprocess the image to enhance its quality for further processing.

        Parameters:
            image_cropped (numpy.ndarray): The cropped image in BGR format.

        Returns:
            tuple: A tuple containing the preprocessed image and an intermediate processed image.
        """
    #Grayscale
    gray_image = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)

    #Histogram Equalization
    equalized_image = cv2.equalizeHist(gray_image)

    #Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(equalized_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101,
                                            2)

    #Morphological Dilation
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(adaptive_thresh, kernel, iterations=1)

    #Morphological Closing
    kernel = np.ones((3, 3), np.uint8)
    cleaning = cv2.morphologyEx(dilated_image, cv2.MORPH_CLOSE, kernel, iterations=4)

    #Median Blur
    median_blurred = cv2.medianBlur(cleaning, 17)

    #Morphological Closing 2
    kernel = np.ones((20, 20), np.uint8)
    cleaning2 = cv2.morphologyEx(median_blurred, cv2.MORPH_CLOSE, kernel, iterations=1)

    #Median Blur 2
    median_blurred2 = cv2.medianBlur(cleaning2, 17)

    return median_blurred2,median_blurred

################################################## READ IMAGE #####################################################

def digit_found(text):
    """
       Check if the text read from the image contains valid digits.

       Parameters:
           text (list): A list of tuples containing the text read from the image and its bounding box coordinates.

       Returns:
           bool: True if the text contains valid digits, False otherwise.
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
            OriginalAdaptTresh (numpy.ndarray): The original image in BGR format.
            ProcessedImage (numpy.ndarray): The processed image.

        Returns:
            tuple: A tuple containing the text read from the image, a boolean indicating whether digits are detected, and the processed image.
        """
    digit_detected = False
    newImage=ProcessedImage
    #READING
    reader = easyocr.Reader(['en'], gpu=True)
    #Read text from the image
    text_ = reader.readtext(newImage)


    for i in range (0,11):
        # Flag to track if any digits are detected
        if not digit_found(text_):

            #Use the processed image for further processing by using different parameters
            kernelSize=22-2*i
            kernel = np.ones((kernelSize, kernelSize), np.uint8)
            cleaning2 = cv2.morphologyEx(OriginalAdaptTresh, cv2.MORPH_CLOSE, kernel, iterations=1)

            #Median blur
            median_blurred2 = cv2.medianBlur(cleaning2, 21)

            #Morphological rectification
            ksize=5
            kernel = np.ones((ksize, ksize), np.uint8)
            rect = cv2.morphologyEx(median_blurred2, cv2.MORPH_RECT, kernel, iterations=1)

            newImage=rect
            #Read the new image

            text_ = reader.readtext(newImage)
        else:
            digit_detected = True
            return text_, digit_detected, newImage

    return text_,digit_detected,newImage