# -*- coding: utf-8 -*-
'''
Cloned from https://github.com/vardanagarwal/Proctoring-AI

@author Jack Kotheimer
@date 2021-07-11
'''

import cv2
import numpy as np
from faceDetector import getFaceDetector, findFaces
from faceLandmarks import getLandmarkModel, detectMarks

'''
Create ROI on mask of the size of eyes and also find the extreme points of each eye

@param (np.uint8) mask - Blank mask to draw eyes on
@param (list<int>) side - The facial landmark numbers of eyes
@param (list<uint32>) shape - Facial landmarks
@return (np.uint8[]) mask - Mask with region of interest drawn
    [l, t, r, b]: left, top, right, and bottom most points of ROI

'''
def getEyeOnMask(mask, side, shape):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    left = points[0][0]
    top = (points[1][1]+points[2][1])//2
    right = points[3][0]
    bottom = (points[4][1]+points[5][1])//2
    return [left, top, right, bottom]

# This is currently based off me measuring distances with a measuring tape while staring at the camera
def getDistance(eyeWidth):
    return 1300/eyeWidth;

 # These magic numbers were here when I got here. Don't ask me, just accept it
left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

# Start capturing video
cap = cv2.VideoCapture(0)

while(True):

    # Get an image and have cv2 draw a square around each face
    ret, img = cap.read()
    faces = findFaces(img, getFaceDetector())
    
    for face in faces:
        # Have tensorflow and cv2 mark the important parts of the face
        shape = detectMarks(img, getLandmarkModel(), face)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # Get arrays describing the edges of each eye : [rightX, topY, leftX, bottomY]
        endPointsLeft = getEyeOnMask(mask, left, shape)
        endPointsRight = getEyeOnMask(mask, right, shape)

        # Take an average of the position of each edge to locate the center of each eye
        leftX = int((endPointsLeft[0] + endPointsLeft[2]) / 2)
        leftY = int((endPointsLeft[1] + endPointsLeft[3]) / 2)
        rightX = int((endPointsRight[0] + endPointsRight[2]) / 2)
        rightY = int((endPointsRight[1] + endPointsRight[3]) / 2)

        # Draw circles around each eye (take this out in prod)
        cv2.circle(img, (leftX, leftY), 4, (0, 0, 255), 2)
        cv2.circle(img, (rightX, rightY), 4, (0, 0, 255), 2)

        # Get an estimated distance from the camera, based on the distance between the eyes
        z = round(getDistance(rightX - leftX), 2);
        print('distance from cam: {} inches'.format(z));
        

    # Show the live view of the camera
    cv2.imshow('eyes', img)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Gracefully exit
cap.release()
cv2.destroyAllWindows()
