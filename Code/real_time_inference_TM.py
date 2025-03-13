
"""
    ** Hand tracker **
    Use cvzone's HandDetector to track one hand at a time, drawing the landmarks and the bounding box
    of the detected hand. Image is further manipulated by modifying the aspect ratio so that it always
    fits in a square sized (imgSize x imgSize), useful for image collection for model training.
    Close the camera by pressing the 0 key.

    Warning: Code crashes when the hand limits are outside the image.

    ** Inference **
    Use Pytorch ANN model to predict the user's sign.
    Can interpret between the adapted ASL signs of "hello", "iloveyou", "no", "sorry", "yes".

    Code adapted from cvzone. Module and examples can be found in:
    https://github.com/cvzone/cvzone?tab=readme-ov-file

"""

import cv2
import math
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = HandDetector(maxHands = 1)
print("Detector loaded.")

offset = 20
imgSize = 224

# labels = ["yes", "iloveyou", "hello", "no", "sorry"]
labels = ["yes", "iloveyou", "no", "sorry", "hello"]
filepath = "C:/Users/visha/OneDrive/Documentos/Code/endor/sign_language/TM_Model"
# classifier = Classifier(f"{filepath}/keras_model.h5", f"{filepath}/labels.txt")
classifier = Classifier(f"{filepath}/keras_model_2.h5", f"{filepath}/labels_2.txt")
print("Classifier loaded.")

while True:
    success, img = cap.read()
    imgOP = img.copy()
    hands, img = detector.findHands(img, draw=True)

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

        pred, index = classifier.getPrediction(imgWhite, draw=True)
        confidence_score = pred[index]
        confidence_score = int(100*confidence_score)

        if confidence_score > 35:
            cv2.rectangle(imgOP, (x-offset,y-offset-30), (x-offset+150,y-offset), (114,57,0), cv2.FILLED)
            cv2.putText(imgOP, labels[index], (x,y-26), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255),2)
            cv2.putText(imgOP, f"Confidence score: {confidence_score}%", (20,40), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255,255,255),2)
            cv2.rectangle(imgOP, (x-offset,y-offset), (x+w+offset,y+h+offset), (114,57,0), 4)
        else:
            cv2.rectangle(imgOP, (x-offset,y-offset), (x+w+offset,y+h+offset), (0,0,255), 4)

        cv2.imshow("Image", imgOP)
        
    else:
        # Create white background
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        cv2.imshow("Image", imgOP)
    
    if cv2.waitKey(1) == ord('0'):
        break

# Gracefully close the program
cap.release()
cv2.destroyAllWindows()