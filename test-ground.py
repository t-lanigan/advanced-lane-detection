from distortion_corrector import distortionCorrector 
import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np


TEST_FOLDER = './camera_cal/'
CAL_IMAGES = TEST_FOLDER + 'calibration*.jpg'
TEST_IMAGE = 'test-cal.jpg'


test = distortionCorrector()

fname = TEST_FOLDER + TEST_IMAGE
img = cv2.imread(fname)
img = img[...,::-1] #convert from opencv bgr to standard rgb

undist = test.undistort(img)

plt.imshow(undist)

plt.show()