import cv2
import numpy as np
import matplotlib.pyplot as plt



def cropImage(im):
    image = im
    if image is None:
        print("Error: Image not found at the specified path.")
        return None, False

    cropSucceed = False

    # Display the original image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original image")
    plt.axis('off')
    plt.show()

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)

    # Visualize the thresholded image
    plt.imshow(thresh, cmap='gray')
    plt.title("Adaptive Threshold")
    plt.axis('off')
    plt.show()

    # Find contours from the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assuming it's the screen)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop the image to only include the screen region
        cropped_image = image[y:y + h, x:x + w]

        # Display the cropped image
        plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        plt.title("Cropped image")
        plt.axis('off')
        plt.show()

        cropSucceed = True
        return cropped_image, cropSucceed
    else:
        print("NO SCREEN DETECTED")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Cannot crop the image")
        plt.axis('off')
        plt.show()
        return image, cropSucceed

# Define the path to your image
#IMAGE_PATH = '../../../Photos/Aircond/AC (7).jpg'
# Test the function with the defined image path
#cropped, bol = cropImage(IMAGE_PATH)