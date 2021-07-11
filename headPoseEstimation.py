# -*- coding: utf-8 -*-
'''
Created on Fri Jul 31 03:00:36 2020

@author: hp
'''

import cv2
import numpy as np
import math
from faceDetector import getFaceDetector, findFaces
from faceLandmarks import getLandmarkModel, detectMarks

'''Return the 3D points present as 2D for making annotation box'''
def get2dPoints(img, rotationVector, translationVector, cameraMatrix, val):

    point3d = []
    distCoeffs = np.zeros((4,1))
    rearSize = val[0]
    rearDepth = val[1]
    point3d.append((-rearSize, -rearSize, rearDepth))
    point3d.append((-rearSize, rearSize, rearDepth))
    point3d.append((rearSize, rearSize, rearDepth))
    point3d.append((rearSize, -rearSize, rearDepth))
    point3d.append((-rearSize, -rearSize, rearDepth))
    
    frontSize = val[2]
    frontDepth = val[3]
    point3d.append((-frontSize, -frontSize, frontDepth))
    point3d.append((-frontSize, frontSize, frontDepth))
    point3d.append((frontSize, frontSize, frontDepth))
    point3d.append((frontSize, -frontSize, frontDepth))
    point3d.append((-frontSize, -frontSize, frontDepth))
    point3d = np.array(point3d, dtype=np.float).reshape(-1, 3)
    
    # Map to 2d img points
    (point2d, _) = cv2.projectPoints(point3d,
                                      rotationVector,
                                      translationVector,
                                      cameraMatrix,
                                      distCoeffs)
    point2d = np.int32(point2d.reshape(-1, 2))
    return point2d

'''
Draw a 3D anotation box on the face for head pose estimation

Parameters
----------
img : np.unit8
    Original Image.
rotationVector : Array of float64
    Rotation Vector obtained from cv2.solvePnP
translationVector : Array of float64
    Translation Vector obtained from cv2.solvePnP
cameraMatrix : Array of float64
    The camera matrix
rearSize : int, optional
    Size of rear box. The default is 300.
rearDepth : int, optional
    The default is 0.
frontSize : int, optional
    Size of front box. The default is 500.
frontDepth : int, optional
    Front depth. The default is 400.
color : tuple, optional
    The color with which to draw annotation box. The default is (255, 255, 0).
lineWidth : int, optional
    line width of lines drawn. The default is 2.

Returns
-------
None.

'''
def draw_annotation_box(img, rotationVector, translationVector, cameraMatrix,
                        rearSize=300, rearDepth=0, frontSize=500, frontDepth=400,
                        color=(255, 255, 0), lineWidth=2):
    
    rearSize = 1
    rearDepth = 0
    frontSize = img.shape[1]
    frontDepth = frontSize*2
    val = [rearSize, rearDepth, frontSize, frontDepth]
    point2d = get2dPoints(img, rotationVector, translationVector, cameraMatrix, val)
    # # Draw all the lines
    cv2.polylines(img, [point2d], True, color, lineWidth, cv2.LINE_AA)
    cv2.line(img, tuple(point2d[1]), tuple(
        point2d[6]), color, lineWidth, cv2.LINE_AA)
    cv2.line(img, tuple(point2d[2]), tuple(
        point2d[7]), color, lineWidth, cv2.LINE_AA)
    cv2.line(img, tuple(point2d[3]), tuple(
        point2d[8]), color, lineWidth, cv2.LINE_AA)
    
'''
Get the points to estimate head pose sideways    

Parameters
----------
img : np.unit8
    Original Image.
rotationVector : Array of float64
    Rotation Vector obtained from cv2.solvePnP
translationVector : Array of float64
    Translation Vector obtained from cv2.solvePnP
cameraMatrix : Array of float64
    The camera matrix

Returns
-------
(x, y) : tuple
    Coordinates of line to estimate head pose

'''
def headPosePoints(img, rotationVector, translationVector, cameraMatrix):

    rearSize = 1
    rearDepth = 0
    frontSize = img.shape[1]
    frontDepth = frontSize*2
    val = [rearSize, rearDepth, frontSize, frontDepth]
    point2d = get2dPoints(img, rotationVector, translationVector, cameraMatrix, val)
    y = (point2d[5] + point2d[8])//2
    x = point2d[2]
    
    return (x, y)
    
face_model = getFaceDetector()
landmarkModel = getLandmarkModel()
cap = cv2.VideoCapture(0)
ret, img = cap.read()
size = img.shape
font = cv2.FONT_HERSHEY_SIMPLEX 
# 3D model points.
modelPoints = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])

# Camera internals
focalLength = size[1]
center = (size[1]/2, size[0]/2)
cameraMatrix = np.array(
                         [[focalLength, 0, center[0]],
                         [0, focalLength, center[1]],
                         [0, 0, 1]], dtype = 'double'
                         )
while True:
    ret, img = cap.read()
    if ret == True:
        faces = findFaces(img, face_model)
        for face in faces:
            marks = detectMarks(img, landmarkModel, face)
            # mark_detector.draw_marks(img, marks, color=(0, 255, 0))
            imagePoints = np.array([
                                    marks[30],     # Nose tip
                                    marks[8],     # Chin
                                    marks[36],     # Left eye left corner
                                    marks[45],     # Right eye right corne
                                    marks[48],     # Left Mouth corner
                                    marks[54]      # Right mouth corner
                                ], dtype='double')
            distCoeffs = np.zeros((4,1)) # Assuming no lens distortion
            (success, rotationVector, translationVector) = cv2.solvePnP(modelPoints, imagePoints, cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_UPNP)
            
            
            # Project a 3D point (0, 0, 1000.0) onto the image plane.
            # We use this to draw a line sticking out of the nose
            
            (noseEndPoint2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotationVector, translationVector, cameraMatrix, distCoeffs)
            
            for p in imagePoints:
                cv2.circle(img, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
            
            
            p1 = ( int(imagePoints[0][0]), int(imagePoints[0][1]))
            p2 = ( int(noseEndPoint2D[0][0][0]), int(noseEndPoint2D[0][0][1]))
            x1, x2 = headPosePoints(img, rotationVector, translationVector, cameraMatrix)

            cv2.line(img, p1, p2, (0, 255, 255), 2)
            cv2.line(img, tuple(x1), tuple(x2), (255, 255, 0), 2)
            # for (x, y) in marks:
            #     cv2.circle(img, (x, y), 4, (255, 255, 0), -1)
            # cv2.putText(img, str(p1), p1, font, 1, (0, 255, 255), 1)
            try:
                m = (p2[1] - p1[1])/(p2[0] - p1[0])
                ang1 = int(math.degrees(math.atan(m)))
            except:
                ang1 = 90
                
            try:
                m = (x2[1] - x1[1])/(x2[0] - x1[0])
                ang2 = int(math.degrees(math.atan(-1/m)))
            except:
                ang2 = 90
                
                # print('div by zero error')
            if ang1 >= 48:
                print('Head down')
                cv2.putText(img, 'Head down', (30, 30), font, 2, (255, 255, 128), 3)
            elif ang1 <= -48:
                print('Head up')
                cv2.putText(img, 'Head up', (30, 30), font, 2, (255, 255, 128), 3)
             
            if ang2 >= 48:
                print('Head right')
                cv2.putText(img, 'Head right', (90, 30), font, 2, (255, 255, 128), 3)
            elif ang2 <= -48:
                print('Head left')
                cv2.putText(img, 'Head left', (90, 30), font, 2, (255, 255, 128), 3)
            
            cv2.putText(img, str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
            cv2.putText(img, str(ang2), tuple(x1), font, 2, (255, 255, 128), 3)
        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cv2.destroyAllWindows()
cap.release()
