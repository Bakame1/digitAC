import cv2
import matplotlib.pyplot as plt
from readImage import readImage
from crop import cropImage

# Load image
image = cv2.imread('../../../Photos/Aircond/AC (7).jpg')

# Pre-process the image
image_cropTab = cropImage(image)
image_cropped, cropSucceed = image_cropTab[0], image_cropTab[1]

# Apply a bilateral blur to the cropped image to the details
blurred_image = cv2.bilateralFilter(image_cropped, 50, 75, 75)

# Display the image with detected digits highlighted
plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
plt.title("Blurred Image")
plt.axis("off")
plt.show()

# Use the blurred image for further processing
image_processed = blurred_image

txtRes = "empty"
################################## READ THE IMAGE WITH EASYOCR ###################################
if cropSucceed:
    # READING
    text_, digit_detected, image_processed = readImage(image_cropped, image_processed)

    if digit_detected and text_ != []:
        txtRes = text_[0][1]

    if digit_detected:
        # If we have more than 3 detected digits
        if len(txtRes) > 3:
            # Remove the last digit if it’s a "0" usually it is a the Celsius letter
            if txtRes[-1] == '0':
                txtRes = txtRes[0] + txtRes[1]  # Remove the last detected "0"

                # Case of S instead of 5
                if txtRes[1] == "S":
                    txtRes = txtRes[0] + "5"

            ####################Draw bounding boxes#############################
            if len(text_) >= 1:
                bbox = text_[0][0]  # Get the bounding box of the detected text area

                # Calculate the approximate width of each digit and the width of the first two digits
                digit_widths = [(bbox[1][0] - bbox[0][0]) / len(text_[0][1])]  # Average width per character
                first_two_digits_width = int(digit_widths[0] * 2)  # Width for first two characters only

                # The bottom right corner of the rectangle we want to draw
                adjusted_bottom_right = (int(bbox[0][0] + first_two_digits_width), int(bbox[2][1]))

                # Draw the rectangle with adjusted width to cover only the first two digits
                cv2.rectangle(image_cropped, tuple(map(int, bbox[0])), adjusted_bottom_right, (0, 255, 0), 2)

                # Display txtRes within the adjusted bounding box (only the first two digits)
                cv2.putText(image_cropped, txtRes[:2], tuple(map(int, bbox[0])), cv2.FONT_HERSHEY_COMPLEX, 4,
                            (255, 0, 0), 4)


        else:
            if len(txtRes) == 2:
                # Case of S instead of 5
                if txtRes[1] == "S":
                    txtRes = txtRes[0] + "5"

            ###############################Draw bounding boxes############################
            bbox = text_[0][0]  # Get the bounding box of the detected text area
            cv2.rectangle(image_cropped, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 0), 5)
            cv2.putText(image_cropped, txtRes, tuple(map(int, bbox[0])), cv2.FONT_HERSHEY_COMPLEX, 4, (255, 0, 0), 4)


        # Display the image with detected digits highlighted
        plt.imshow(cv2.cvtColor(image_cropped, cv2.COLOR_BGR2RGB))
        plt.title("Digits found")
        plt.axis("off")
        plt.show()
    else:
        print("No digit detected")
else :
    print("Cant crop")
