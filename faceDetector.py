# -*- coding: utf-8 -*-
'''
Created on Wed Jul 29 17:52:00 2020

@author: hp
'''

import cv2
import numpy as np

# Get a quantized tensorflow face detection model
def getFaceDetector(modelFile='models/opencv_face_detector_uint8.pb', configFile='models/opencv_face_detector.pbtxt'):
    return cv2.dnn.readNetFromTensorflow(modelFile, configFile)

'''
Find the faces in an image

@param (np.uint8) img - Image to find faces from
@param (dnn_Net) model - Face detection model
@return (list<[right, top, left, bottom]>) faces - List of boxes of the faces detected in the image
'''
def findFaces(img, model):
    faces = []

	# cv2 magic
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )
    model.setInput(blob)
    res = model.forward()
    height, width = img.shape[:2]
    for i in range(res.shape[2]):
        confidence = res[0, 0, i, 2]
        if confidence > 0.4:
            box = res[0, 0, i, 3:7] * np.array([width, height, width, height])
            (right, top, left, bottom) = box.astype('int')
            faces.append([right, top, left, bottom])
    return faces
