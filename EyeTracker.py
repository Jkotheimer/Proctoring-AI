# -*- coding: utf-8 -*-
'''
Cloned from https://github.com/vardanagarwal/Proctoring-AI

@author Jack Kotheimer
@date 2021-07-11
'''
import numpy as np

'''
Create ROI on mask of the size of eyes and also find the extreme points of each eye

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

# This is currently based off me measuring distances with a measuring tape while staring at the camera
def getDistanceBetweenEyes(eyeWidth):
    return 1300/eyeWidth;

 # These magic numbers were here when I got here. Don't ask me, just accept it
left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

def getEyes(shape):

    # Get arrays describing the edges of each eye : [rightX, topY, leftX, bottomY]
    endPointsLeft = getEye(left, shape)
    endPointsRight = getEye(right, shape)

    # Take an average of the position of each edge to locate the center of each eye
    leftX = int((endPointsLeft[0] + endPointsLeft[2]) / 2)
    leftY = int((endPointsLeft[1] + endPointsLeft[3]) / 2)
    rightX = int((endPointsRight[0] + endPointsRight[2]) / 2)
    rightY = int((endPointsRight[1] + endPointsRight[3]) / 2)

    # Get an estimated distance from the camera, based on the distance between the eyes
    z = int(getDistanceBetweenEyes(rightX - leftX));

    return [(leftX, leftY, z), (rightX, rightY, z)]

def getEyesFast(face):
    # face: [left, top, right, bottom]
    #       [+x, +y, -x, -y]
    left = face[0]
    top = face[1]
    right = face[2]
    bottom = face[3]

    y = int(top + ((bottom - top) * 5 / 12))
    lx = int(left + ((right - left) / 4))
    rx = int(right - ((right - left) / 4))
    z = int(getDistanceBetweenEyes(rx - lx));
    return [(lx, y, z), (rx, y, z)]
