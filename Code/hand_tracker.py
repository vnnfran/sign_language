
"""
    ** Hand tracker **
    Use cvzone's HandDetector to track one hand at a time, drawing the landmarks and the bounding box
    of the detected hand. Image is further manipulated by modifying the aspect ratio so that it always
    fits in a square sized (imgSize x imgSize), useful for image collection for model training.
    Close the camera by pressing the 0 key.

    Warning: Code crashes when the hand limits are outside the image.

    Code adapted from cvzone. Module and examples can be found in:
    https://github.com/cvzone/cvzone?tab=readme-ov-file

"""

import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands = 1)
offset = 20
imgSize = 300

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img, draw=True, flipType=True)

    if hands:
        # If hand is detected, get the coords of its bounding box
        hand = hands[0]
        x,y,w,h = hand['bbox']

        # Create white background
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        # Crop the bounding box
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        imgCropShape = imgCrop.shape
        aspectRatio = h/w

        # Reshape the bounding box so it fits in the square
        if aspectRatio > 1:
            k = imgSize/h
            wcal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wcal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wcal)/2)
            imgWhite[:, wGap:wcal+wGap] = imgResize
        else:
            k = imgSize/w
            hcal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop, (hcal, imgSize))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hcal)/2)
            imgWhite[:, hGap:hcal+hGap] = imgResize

        cv2.imshow("Crop", imgCrop)
        cv2.imshow("White", imgWhite)
   
    else:
        # Create white background
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        cv2.imshow("White", imgWhite)

    cv2.imshow("Image", img)
    
    if cv2.waitKey(1) == ord('0'):
        break

# Gracefully close the program
cap.release()
cv2.destroyAllWindows()