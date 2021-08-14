'''
Cloned from https://github.com/vardanagarwal/Proctoring-AI

@author Jack Kotheimer
@date 2021-07-11
'''

import cv2
import json
import numpy as np
from FaceLandmarks import getLandmarkModel, detectMarks

'''
getEye
------
@param (list<int>) side - The facial landmark numbers of eyes
@param (list<uint32>) shape - Facial landmarks
@return (np.uint8[]) mask - Mask with region of interest drawn
    [l, t, r, b]: left, top, right, and bottom most points of ROI

'''
def getEye(side, shape):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    left = points[0][0]
    top = (points[1][1]+points[2][1])//2
    right = points[3][0]
    bottom = (points[4][1]+points[5][1])//2
    return [left, top, right, bottom]

'''
getEyes
-------
Retreive the pixel locations of both eyes on a face

@param face (uint8_t[]): [left, top, right, bottom] edges of the face. This is just a square region, in which the face resides
@param fast (bool): Use the fast algorithm to save time at the expense of accuracy
@param model (optional)(string): path/to/model.file only used if fast == False, a default is provided in FaceLandmarks.py

@returns eyes (tuple(x,y,z)[]): [(lx,ly,lz), (rx,ry,rz)] coordinates 
'''
def getEyes(img, face, model):

    # Get the most important landmarks of the face
    shape = detectMarks(img, getLandmarkModel(model), face)

    # These magic numbers were here when I got here. Don't ask me, just accept it
    leftMagic = [36, 37, 38, 39, 40, 41]
    rightMagic = [42, 43, 44, 45, 46, 47]

    # Get arrays describing the edges of each eye : [rightX, topY, leftX, bottomY]
    endPointsLeft = getEye(leftMagic, shape)
    endPointsRight = getEye(rightMagic, shape)

    # Take an average of the position of each edge to locate the center of each eye
    Lx = int((endPointsLeft[0] + endPointsLeft[2]) / 2)
    Ly = int((endPointsLeft[1] + endPointsLeft[3]) / 2)
    Rx = int((endPointsRight[0] + endPointsRight[2]) / 2)
    Ry = int((endPointsRight[1] + endPointsRight[3]) / 2)

    # Get an estimated distance from the camera, based on the distance between the eyes
    z = int(getDistanceBetweenEyes(Rx - Lx));

    return [(Lx, Ly, z), (Rx, Ry, z)]


'''
Instead of describing the math, lemme give you a visual...

         Lx    Rx   1/4 and 3/4 width (relatively where the eyes are, width wise, on the face)
         |_____|___/
         |     |
         |     |
     |---|-----|----|
     |   |     |    |
     |   |     |    |
     |   x-----x----------y - 5/12 height (relatively where the eyes are, height wise, on the face)
     |              |
     |      <       | <-----face box
     |              |
     |    \____/    |
     |              |
     |______________|

*** This is meant to be barely accurate, yet fast as hell ***
*** If you're wondering how I came up with these numbers, I just looked at my face at tweaked it until it looked correct lol ***

'''
def getEyesFast(face):
    # face: [left, top, right, bottom]
    #       [+x, +y, -x, -y]
    left = face[0]
    top = face[1]
    right = face[2]
    bottom = face[3]

    Lx = int(left + ((right - left) / 4))
    Rx = int(right - ((right - left) / 4))
    y = int(top + ((bottom - top) * 5 / 12))
    z = int(getDistanceBetweenEyes(Rx - Lx));
    return [(Lx, y, z), (Rx, y, z)]

# This is currently based off me measuring distances with a measuring tape while staring at the camera
def getDistanceBetweenEyes(eyeWidth):
    return 1300/eyeWidth;

def drawEyes(img, eyes):
    cv2.circle(img, (eyes[0][0], eyes[0][1]), 4, (0, 0, 255), 2)
    cv2.circle(img, (eyes[1][0], eyes[1][1]), 4, (0, 0, 255), 2)
