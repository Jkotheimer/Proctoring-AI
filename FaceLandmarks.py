# -*- coding: utf-8 -*-
'''
Created on Wed Jul 29 19:47:08 2020

@author: hp
'''

import cv2
import numpy as np
import tensorflow as tf

'''
Get the facial landmark model. 
Original repository: https://github.com/yinguobing/cnn-facial-landmark

@param (string, optional) savedModel - Path to facial landmarks model. The default is 'models/pose_model'.
@return (tf model) model - Facial landmarks model
'''
def getLandmarkModel(savedModel='models/pose_model'):
    return tf.saved_model.load(savedModel)

'''Get a square box out of the given box, by expanding it.'''
def getSquareBox(box):
    left_x = box[0]
    top_y = box[1]
    right_x = box[2]
    bottom_y = box[3]

    box_width = right_x - left_x
    box_height = bottom_y - top_y

    # Check if box is already a square. If not, make it a square.
    diff = box_height - box_width
    delta = int(abs(diff) / 2)

    if diff == 0:                  # Already a square.
        return box
    elif diff > 0:                # Height > width, a slim box.
        left_x -= delta
        right_x += delta
        if diff % 2 == 1:
            right_x += 1
    else:                          # Width > height, a short box.
        top_y -= delta
        bottom_y += delta
        if diff % 2 == 1:
            bottom_y += 1

    # Make sure box is always square.
    assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'

    return [left_x, top_y, right_x, bottom_y]

'''Move the box to direction specified by vector offset'''
def moveBox(box, offset):
    left_x = box[0] + offset[0]
    top_y = box[1] + offset[1]
    right_x = box[2] + offset[0]
    bottom_y = box[3] + offset[1]
    return [left_x, top_y, right_x, bottom_y]

'''
Find the facial landmarks in an image from the faces

@param (np.uint8) img - The image in which landmarks are to be found
@param (tf model) model - Loaded facial landmark model
@param (list) face - Face coordinates (x, y, x1, y1) in which the landmarks are to be found
@return (np.array) marks - Facial landmark points
'''
def detectMarks(img, model, face):
    offset_y = int(abs((face[3] - face[1]) * 0.1))
    boxMoved = moveBox(face, [0, offset_y])
    faceBox = getSquareBox(boxMoved)
    
    h, w = img.shape[:2]
    if faceBox[0] < 0:
        faceBox[0] = 0
    if faceBox[1] < 0:
        faceBox[1] = 0
    if faceBox[2] > w:
        faceBox[2] = w
    if faceBox[3] > h:
        faceBox[3] = h
    
    faceImg = img[faceBox[1]: faceBox[3], faceBox[0]: faceBox[2]]
    faceImg = cv2.resize(faceImg, (128,128))
    faceImg = cv2.cvtColor(faceImg, cv2.COLOR_BGR2RGB)
    
    # # Actual detection.
    predictions = model.signatures['predict'](
        tf.constant([faceImg], dtype=tf.uint8))

    # Convert predictions to landmarks.
    marks = np.array(predictions['output']).flatten()[:136]
    marks = np.reshape(marks, (-1, 2))
    
    marks *= (faceBox[2] - faceBox[0])
    marks[:, 0] += faceBox[0]
    marks[:, 1] += faceBox[1]
    marks = marks.astype(np.uint)

    return marks

'''
Draw the facial landmarks on an image

@param (np.uint8) image - Image on which landmarks are to be drawn.
@param (np.array) marks - Facial landmark points
@param (tuple, optional) color - Color to which landmarks are to be drawn with. The default is (0, 255, 0).
@return void
'''
def drawMarks(image, marks, color=(0, 255, 0)):
    for mark in marks:
        cv2.circle(image, (mark[0], mark[1]), 2, color, -1, cv2.LINE_AA)
    
