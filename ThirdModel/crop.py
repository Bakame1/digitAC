import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    #If incorrect path
    if image is None:
        print("Error: Image not found at the specified path.")
        return None, False

    #Convert the image to grayscale + gamma correction
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_gamma = apply_gamma_correction(gray, gamma=2.2)

    #Display the gamma-corrected image
    plt.imshow(gray_gamma, cmap='gray')
    plt.title("Gamma Corrected Image")
    plt.axis('off')
    plt.show()

    #Apply adaptive thresholding with different block sizes
    block_sizes = [21, 51, 81]
    C = 2
    largest_cropped_image = None
    largest_width = 0

    for block_size in block_sizes:
        thresh = cv2.adaptiveThreshold(gray_gamma, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, block_size, C)

        """
        plt.imshow(thresh, cmap='gray')
        plt.title(f"Adaptive Threshold with block size {block_size}")
        plt.axis('off')
        plt.show()"""

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cropped_image = gray_gamma[y:y + h, x:x + w]

            #Select the largest width between these different thresholding
            if w > largest_width:
                largest_width = w
                largest_cropped_image = cropped_image

    if largest_cropped_image is not None:
        plt.imshow(cv2.cvtColor(largest_cropped_image, cv2.COLOR_BGR2RGB))
        plt.title("Largest Cropped image by Width")
        plt.axis('off')
        plt.show()
        return largest_cropped_image, True
    else:
        print("NO SCREEN DETECTED")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Cannot crop the image")
        plt.axis('off')
        plt.show()
        return image, False

IMAGE_PATH = '../../../Photos/Aircond/AC (7).jpg'
image=cv2.imread(IMAGE_PATH)
cropped, bol = cropImage(image)
