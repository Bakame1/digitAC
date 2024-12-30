import cv2
import numpy as np
from matplotlib import pyplot as plt

def cropImage(im):
    """
        Crop the image to extract the region of interest.

        Parameters:
            im (numpy.ndarray): The input image in BGR format.

        Returns:
            numpy.ndarray: The cropped image.
        """
    image = im
    #Display the original image
    plt.imshow(image)
    plt.title("Original image")
    plt.axis('off')
    plt.show()

    #HSV conversion
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #Define threshold colors (gray and green)
    lower_bound = np.array([30, 10, 10])   # Tint, Saturation, minimal value
    upper_bound = np.array([90, 50, 200])

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

        #Display the cropped image
        plt.imshow(cropped_image)
        plt.title("Cropped image")
        plt.axis('off')
        plt.show()

    else:
        print("NO SCREEN DETECTED")

    return cropped_image

#Test
#Load image
test = cv2.imread('../../../Photos/Aircond/AC (11).jpg')

# Appliquer la fonction de rognage de l'image
cropImage(test)
