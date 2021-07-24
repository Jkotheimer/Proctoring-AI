'''
Cloned from https://github.com/vardanagarwal/Proctoring-AI

@author Jack Kotheimer
@date 2021-07-11
'''
import cv2
import time
import sys
from FaceDetector import getFaceDetector, findFace, drawFace
from FaceLandmarks import getLandmarkModel, detectMarks
from EyeTracker import getEyes, getEyesFast

# Start capturing video
cap = cv2.VideoCapture(0)

fast = 'fast' in sys.argv
display = 'display' in sys.argv

if display:
    cv2.namedWindow('preview')

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
    while(True):

        # Capture image from camera
        # -------------------------
        tstamp = time.time() * 1000
        ret, img = cap.read()
        if not ret:
            print('IMAGE NOT CAPTURED')
            continue
        duration = (time.time() * 1000) - tstamp
        print('image capture took {} ms'.format(round(duration, 4)))
        tstamp = time.time() * 1000
    
        # Get a square region around the face
        # -----------------------------------
        face = findFace(img, getFaceDetector())
        if len(face) == 0:
            print('FACE NOT DETECTED')
            continue
        duration = (time.time() * 1000) - tstamp
        print('face detection took {} ms'.format(round(duration, 2)))
        tstamp = time.time() * 1000
    
        # Determine the location of each eye
        # eyes: [(xL, yL, zL), (xR, yR, zR)]
        if fast:
            eyes = getEyesFast(face)
        else:
            shape = detectMarks(img, getLandmarkModel(), face)
            if len(shape) == 0:
                print('SHAPE NOT DETECTED')
                continue
    
            eyes = getEyes(shape)
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

if __name__ == "__main__":
    run()
