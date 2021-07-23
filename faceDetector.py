# -*- coding: utf-8 -*-
'''
Created on Wed Jul 29 17:52:00 2020

@author: hp
'''

import cv2
import time
import gc
import numpy as np

# Get a quantized tensorflow face detection model
def getFaceDetector(modelFile='models/opencv_face_detector_uint8.pb', configFile='models/opencv_face_detector.pbtxt'):
    return cv2.dnn.readNetFromTensorflow(modelFile, configFile)

'''
Find the faces in an image

@param (np.uint8) img - Image to find faces from
@param (dnn_Net) model - Face detection model
@return ([right, top, right, bottom]) faces - The box around the most confident face in the image
'''
def findFace(img, model):

    # cv2 magic
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )
    model.setInput(blob)
    res = model.forward()

    # Iterate over everything the model matched with and return the guess with the highest confidence
    face = []
    highestConfidence = 0
    for i in range(res.shape[2]):
        confidence = res[0, 0, i, 2]
        if confidence > 0.5 and confidence > highestConfidence:
            box = res[0, 0, i, 3:7] * np.array([width, height, width, height])
            # [+x, +y, -x, -y]
            (left, top, right, bottom) = box.astype('int')
            face = [left, top, right, bottom]
            highestConfidence = confidence

    del res
    del model
    del blob
    gc.collect()

    return face

def drawFace(img, face):
    cv2.rectangle(img, (face[0], face[1]), (face[2], face[3]), (0, 0, 255), 3)
