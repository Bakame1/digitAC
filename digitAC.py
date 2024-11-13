from operator import truediv
import imutils
from imutils.perspective import four_point_transform
import cv2
import easyocr
import matplotlib.pyplot as plt
import re


#Load image
image = cv2.imread('../../Photos/Aircond/AC (1).jpg')

#Pre-process the image
image = imutils.resize(image, height=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#gray filter

#Increase contrast with histogram equalization
gray = cv2.equalizeHist(gray)

#####################CROP THE IMAGE TO GET A SMALLER AREA##############################
#Apply Gaussian blur and edge detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 200, 255)

#Find contours and sort them by size in descending order
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
displayCnt = None

#Loop over contours to find a rectangular display area
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        displayCnt = approx
        break


##################################Extract the display##############################
image_cropped = four_point_transform(image, displayCnt.reshape(4, 2))

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)
text_ = reader.readtext(image_cropped)

# Set the confidence threshold lower to detect more potential digits
threshold = 0.15

# Regular expression pattern for digits only
pattern = re.compile(r'^\d+$')

# Flag to track if any digits are detected
digit_detected = False


txtCopy=text_[0][1]

#If we have more than 3 detected digits
if len(txtCopy) == 3:
    #Remove the last digit if it’s a "0" usually it is a the Celsius letter
    if txtCopy[-1]== '0':
        txtCopy=txtCopy[0]+txtCopy[1]  # Remove the last detected "0"
        print(txtCopy)

    ####################Draw bounding boxes#############################
    if len(text_) >= 1:
        bbox = text_[0][0]  #Get the bounding box of the detected text area

        #Calculate the approximate width of each digit and the width of the first two digits
        digit_widths = [(bbox[1][0] - bbox[0][0]) / len(text_[0][1])]  # Average width per character
        first_two_digits_width = int(digit_widths[0] * 2)  # Width for first two characters only

        #The bottom right corner of the rectangle we want to draw
        adjusted_bottom_right = (int(bbox[0][0] + first_two_digits_width), int(bbox[2][1]))

        #Draw the rectangle with adjusted width to cover only the first two digits
        cv2.rectangle(image_cropped, tuple(map(int, bbox[0])), adjusted_bottom_right, (0, 255, 0), 2)

        #Display txtCopy within the adjusted bounding box (only the first two digits)
        cv2.putText(image_cropped, txtCopy[:2], tuple(map(int, bbox[0])), cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)

        digit_detected=True
else:
    ###############################Draw bounding boxes############################
    for bbox, text, score in text_:
        if score > threshold and pattern.match(text):  # Check if text contains only digits
            cv2.rectangle(image_cropped, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 0), 5)
            cv2.putText(image_cropped, text, tuple(map(int, bbox[0])), cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)
            digit_detected = True

#No digit exception
if not digit_detected:
    print("NO DIGIT DETECTED")

#Display the image with detected digits highlighted
plt.imshow(cv2.cvtColor(image_cropped, cv2.COLOR_BGR2RGB))
plt.show()

#Display the string value read
print(text_)

