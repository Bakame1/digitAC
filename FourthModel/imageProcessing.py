import cv2
import numpy as np
from matplotlib import pyplot as plt


def preprocessImage(image_cropped):
    """
        Preprocess the input image by applying various image processing techniques.

        Parameters:
            image_cropped (numpy.ndarray): The cropped image in BGR format.

        Returns:
            tuple: A tuple containing the preprocessed image and the median-blurred image.
        """
    #Convert to Grayscale
    gray_image = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)

    #Histogram Equalization
    equalized_image = cv2.equalizeHist(gray_image)

    #Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(equalized_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101,
                                            2)
    plt.imshow(cv2.cvtColor(adaptive_thresh, cv2.COLOR_BGR2RGB))
    plt.title("adaptative thresh")
    plt.axis("off")
    plt.show()

    #Morphological Dilation
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(adaptive_thresh, kernel, iterations=1)
    plt.imshow(cv2.cvtColor(dilated_image, cv2.COLOR_BGR2RGB))
    plt.title("dilated")
    plt.axis("off")
    plt.show()

    #Morphological Closing
    kernel = np.ones((3, 3), np.uint8)
    cleaning = cv2.morphologyEx(dilated_image, cv2.MORPH_CLOSE, kernel, iterations=4)
    plt.imshow(cv2.cvtColor(cleaning, cv2.COLOR_BGR2RGB))
    plt.title("cleaning")
    plt.axis("off")
    plt.show()

    #Median Blur
    median_blurred = cv2.medianBlur(cleaning, 17)
    plt.imshow(cv2.cvtColor(median_blurred, cv2.COLOR_BGR2RGB))
    plt.title("median_blurred")
    plt.axis("off")
    plt.show()

    #Morphological Closing 2
    kernel = np.ones((20, 20), np.uint8)
    cleaning2 = cv2.morphologyEx(median_blurred, cv2.MORPH_CLOSE, kernel, iterations=1)
    plt.imshow(cv2.cvtColor(cleaning2, cv2.COLOR_BGR2RGB))
    plt.title("cleaning2")
    plt.axis("off")
    plt.show()

    #Median Blur 2
    median_blurred2 = cv2.medianBlur(cleaning2, 17)
    plt.imshow(cv2.cvtColor(median_blurred2, cv2.COLOR_BGR2RGB))
    plt.title("median_blurred2")
    plt.axis("off")
    plt.show()


    return median_blurred2,median_blurred