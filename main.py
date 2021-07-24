# -*- coding: utf-8 -*-
'''
Cloned from https://github.com/vardanagarwal/Proctoring-AI

@author Jack Kotheimer
@date 2021-07-11
'''
import cv2
import time
import sys
from faceDetector import getFaceDetector, findFace, drawFace
from faceLandmarks import getLandmarkModel, detectMarks
from eyeTracker import getEyes, getEyesFast

# Start capturing video
cap = cv2.VideoCapture(0)
#cv2.namedWindow('preview')

args = {
    'fast': fast
}

while(True):
    tstamp = time.time() * 1000
    ret, img = cap.read()

    if not ret:
        print('IMAGE NOT CAPTURED')
        continue

    duration = (time.time() * 1000) - tstamp
    print('image capture took {} ms'.format(round(duration, 4)))
    tstamp = time.time() * 1000

    # Get an image and have cv2 draw a square around each face
    face = findFace(img, getFaceDetector())
    if len(face) == 0:
        print('FACE NOT DETECTED')
        continue
    
    duration = (time.time() * 1000) - tstamp
    print('face detection took {} ms'.format(round(duration, 2)))
    tstamp = time.time() * 1000

    # Have tensorflow and cv2 mark the important parts of the face
    #shape = detectMarks(img, getLandmarkModel(), face)
    #if len(shape) == 0:
    #    print('SHAPE NOT DETECTED')
    #    continue

    # [(left x, y, z), (right x, y, z)]
    #eyes = getEyes(shape)
    eyes = getEyesFast(face)
    print(eyes)

    # THIS BLOCK IS FOR DEBUGGING REASONS
    # -----------------------------------------------------------
    duration = (time.time() * 1000) - tstamp
    print('eye pinpointing took {} ms'.format(round(duration, 2)))
    tstamp = time.time() * 1000
    #cv2.circle(img, (eyes[0][0], eyes[0][1]), 4, (0, 0, 255), 2)
    #cv2.circle(img, (eyes[1][0], eyes[1][1]), 4, (0, 0, 255), 2)
    #drawFace(img, face)
    #cv2.imshow('preview', img)
    #duration = (time.time() * 1000) - tstamp
    #print('displaying took {} ms'.format(round(duration, 2)))
    #tstamp = time.time() * 1000
    # -----------------------------------------------------------

    # Terminate when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Gracefully exit
cap.release()
cv2.destroyAllWindows()
