'''
Cloned from https://github.com/vardanagarwal/Proctoring-AI

@author Jack Kotheimer
@date 2021-07-11
'''
import cv2
import time
import sys
import os.path
import configparser
from FaceDetector import getFaceDetector, findFace, drawFace
from EyeTracker import getEyes

def help():
    print('Valid arguments (python SmartVisor.py <arg>)')
    print('  DEBUG_0: Debug the program with all debugging info, including video preview')
    print('  DEBUG_1: Debug the program with text debugging info only')
    print('  PROD:    Run this program as a production ready product')
    exit(1)

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
if len(sys.argv) != 2:
    print('ERROR: Exactly one argument required')
    help()

configFilename = 'config/config.ini'
config = configparser.ConfigParser()
config.read(configFilename)
if len(config.sections()) == 0:
    print('ERROR: config file {} invalid or not found'.format(configFilename))
    exit(2)

env = sys.argv[1]
fast = config[env].getboolean('fast')
debug = config[env].getboolean('debug')
display = config[env].getboolean('display')

# Start capturing video
cap = cv2.VideoCapture(0)

if display:
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
        if debug:
            duration = (time.time() * 1000) - tstamp
            print('image capture took {} ms'.format(round(duration, 4)))
            tstamp = time.time() * 1000
        # -----------------------------------
        # Get a square region around the face
        # -----------------------------------
        face = findFace(img, getFaceDetector())
        if len(face) == 0:
            print('FACE NOT DETECTED')
            continue
        if debug:
            duration = (time.time() * 1000) - tstamp
            print('face detection took {} ms'.format(round(duration, 2)))
            tstamp = time.time() * 1000
        # ----------------------------------  
        # Determine the location of each eye
        # ----------------------------------
        eyes = getEyes(face, fast) # [(Lx, Ly, Lz), (Rx, Ry, Rz)]
        if debug:
            duration = (time.time() * 1000) - tstamp
            print('face detection took {} ms'.format(round(duration, 2)))
            tstamp = time.time() * 1000
    
        # THIS BLOCK IS FOR DEBUGGING REASONS
        # -----------------------------------------------------------
        duration = (time.time() * 1000) - tstamp
        print('eye pinpointing took {} ms'.format(round(duration, 2)))
        tstamp = time.time() * 1000
        drawFace(img, face)
        drawEyes(img, face)
        #cv2.imshow('preview', img)
        #duration = (time.time() * 1000) - tstamp
        #print('displaying took {} ms'.format(round(duration, 2)))
        #tstamp = time.time() * 1000
        # -----------------------------------------------------------
    
        # Terminate when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('f'):
            config[env]['fast'] = 'no' if fast else 'yes'
            with open(configFilename, 'w') as configFile:
                config.write(configFile)

    
    # Gracefully exit
    cap.release()
    cv2.destroyAllWindows()
# ---------------------------------------------------------------
# / END RUN /
# ---------------------------------------------------------------

if __name__ == "__main__":
    run()
