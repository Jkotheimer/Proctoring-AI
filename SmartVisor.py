'''
Cloned from https://github.com/vardanagarwal/Proctoring-AI

@author Jack Kotheimer
@date 2021-07-11
'''
import cv2
import time
import sys
import json
import os.path
from FaceDetector import getFaceDetector, findFace, drawFace
from EyeTracker import getEyes, getEyesFast, drawEyes
from utils import debugTime

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
configFilename = 'config/config.json'
with open(configFilename) as configFile:
    config = json.load(configFile)

with open(config['camera']) as camLocationFile:
    cameraLocation = json.load(camLocationFile)

with open(config['windshield']) as wsLocationFile:
    windshieldLocation = json.load(wsLocationFile)

print(config)
print(cameraLocation)
print(windshieldLocation)

# Start capturing video
cap = cv2.VideoCapture(0)

if config['display']:
    cv2.namedWindow('preview')

# ---------------------------------------------------
# / END CONFIG /
# ---------------------------------------------------

'''
SmartVisor Main function
Continuously
    - Capture image
    - Detect face and eyes
    - Retrieve radial location of Sun
    - Calculate ray between eyes and sun (rayEyeSun)
    - Retrieve coordinates of car
    - Retrieve cardinal direction of car
    - Determine if rayEyeSun crosses the windshield of the car
    - Post location of cross point (if it exists) to an event listener
'''
def run():
    tstamp = time.time() * 1000
    while(True):
        # -------------------------
        # Capture image from camera
        # -------------------------
        ret, img = cap.read()
        if not ret:
            print('IMAGE NOT CAPTURED')
            continue
        if config['debug']:
            print('----------------------------')
            print('     ACTION     | DURATION ')
            print('----------------------------')
            tstamp = debugTime('Image Capture', tstamp, time.time() * 1000)
        # -----------------------------------
        # Get a square region around the face
        # -----------------------------------
        face = findFace(img, getFaceDetector())
        if len(face) == 0:
            print('FACE NOT FOUND')
            continue

        if config['debug']:
            tstamp = debugTime('Face Detection', tstamp, time.time() * 1000)
        # ----------------------------------  
        # Determine the location of each eye
        # ----------------------------------
        if config['fast']:
            eyes = getEyesFast(face)
        else:
            eyes = getEyes(img, face, 'models/pose_model') # [(Lx, Ly, Lz), (Rx, Ry, Rz)]
        if config['debug']:
            tstamp = debugTime('Eye Detection', tstamp, time.time() * 1000)
    
        if config['display']:
            drawFace(img, face)
            drawEyes(img, eyes)
            cv2.imshow('preview', img)
            if config['debug']:
                tstamp = debugTime('Image Display', tstamp, time.time() * 1000)
    
        # Terminate when 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            print('FAST!!!!!!!')
            config['fast'] = not config['fast']
            with open(configFilename, 'w') as configFile:
                json.dump(config, configFile)

    
    # Gracefully exit
    cap.release()
    cv2.destroyAllWindows()
# ---------------------------------------------------------------
# / END RUN /
# ---------------------------------------------------------------

if __name__ == "__main__":
    run()
