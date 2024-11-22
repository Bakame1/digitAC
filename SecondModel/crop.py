import cv2
import numpy as np
from matplotlib import pyplot as plt

def cropImage(im):
    image = im
    #Display the original image
    plt.imshow(image)
    plt.title("Original image")
    plt.axis('off')
    plt.show()

    # Convertir l'image en HSV pour la segmentation de couleur
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Définir les seuils pour la couleur de l'écran (gris/vert)
    lower_bound = np.array([30, 10, 10])   # Teinte, Saturation, Valeur minimales
    upper_bound = np.array([90, 50, 200])  # Teinte, Saturation, Valeur maximales

    # Créer un masque avec les seuils définis
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Trouver les contours à partir du masque
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Trouver le plus grand contour (supposé être l'écran du thermostat)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Rogner l'image pour garder seulement la région de l'écran
        cropped_image = image[y:y+h, x:x+w]

        #Display the cropped image
        plt.imshow(cropped_image)
        plt.title("Cropped image")
        plt.axis('off')
        plt.show()

    else:
        print("NO SCREEN DETECTED")

    return cropped_image
"""
# Charger l'image
test = cv2.imread('../../../Photos/Aircond/AC (2).jpg')

# Appliquer la fonction de rognage de l'image
cropImage(test)
"""