from collections import deque
#from imutils.video import VideoStream
import numpy as np
#import argparse
import cv2
#import imutils
import time

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
pts = deque(maxlen=64)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 5,
                       qualityLevel = 0.2,
                       minDistance =200,
                       blockSize = 15 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
 
# otherwise, grab a reference to the video file
cap = cv2.VideoCapture("outvid2.mp4")

# allow the camera or video file to warm up
time.sleep(0.5)


clahe = True
rgb = False

# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
points = np.array([[0,0]])
for i in range(60):
    ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
print(p0)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
while(1):
    ret,frame = cap.read()
    if not ret:
        break
    #img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if clahe:
        # -----Converting image to LAB Color model-----------------------------------
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
        l, a, b = cv2.split(lab)
        l = cv2.fastNlMeansDenoising(l, None, 13, 25)


        limg = cv2.merge((l, l, l))

        # -----Converting image from LAB Color model to RGB model--------------------
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        final = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
        cv2.imshow('final', final)

    if rgb:
        b = frame.copy()
        # set green and red channels to 0
        b[:, :, 1] = 0
        b[:, :, 2] = 0

        g = frame.copy()
        # set blue and red channels to 0
        g[:, :, 0] = 0
        g[:, :, 2] = 0

        r =frame.copy()
        # set blue and green channels to 0
        r[:, :, 0] = 0
        r[:, :, 1] = 0

        # RGB - Blue
        cv2.imshow('B-RGB', b)

        # RGB - Green
        cv2.imshow('G-RGB', g)

        # RGB - Red
        cv2.imshow('R-RGB', r)

        cv2.waitKey(0)

    # if the 'q' key is pressed, stop the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
