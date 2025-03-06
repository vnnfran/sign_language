
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

import os
import cv2
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands = 1)
offset = 20
imgSize = 150

### Load model ###
class MultilayerPerceptron(nn.Module):
    def __init__(self, in_sz=22500, out_sz=5, layers=[4500, 900, 180, 36]):
        super().__init__()
        self.fc1 = nn.Linear(in_sz,layers[0])
        self.fc2 = nn.Linear(layers[0],layers[1])
        self.fc3 = nn.Linear(layers[1],layers[2])
        self.fc4 = nn.Linear(layers[2],layers[3])
        self.fc5 = nn.Linear(layers[3],out_sz)

    def forward(self,X):
        X = F.relu(self.fc1(X))           # ReLU as activation function.
        X = F.relu(self.fc2(X))           # ReLU as activation function.
        X = F.relu(self.fc3(X))           # ReLU as activation function.
        X = F.relu(self.fc4(X))           # ReLU as activation function.
        X = self.fc5(X)                   # Direct pass.
        return F.log_softmax(X, dim=1)    # Get class probabilities

model = MultilayerPerceptron()

model_dir = "C:\\Users\\visha\\OneDrive\\Documentos\\Code\\endor\\sign_language\\Model"
model_path = os.path.join(model_dir, "model"+".pth")
model.load_state_dict(torch.load(model_path, weights_only=True))

### Open camera and perform ###
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

        input_img = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY )
        X = np.array(input_img)
        X = X.astype(np.float32)
        X = np.flatten(X)
        print(X.shape)
        #y_pred = model(X.view(X.size(0),-1))
        #predicted = torch.max(y_pred,1)[1]
        # print(predicted)

    else:
        # Create white background
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        # cv2.imshow("White", imgWhite)

    cv2.imshow("Image", img)
    
    if cv2.waitKey(1) == ord('0'):
        break

# Gracefully close the program
cap.release()
cv2.destroyAllWindows()