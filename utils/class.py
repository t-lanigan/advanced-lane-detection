import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils.image_processing import distortionCorrector
import pickle
import os
import glob



class Pipeline(object):    
    
    def __init__(self, calibration_folder_path):
        """Takes in path to calibration files as string"""

        # Set nx and ny according to how many inside corners in chess boards.
        self.var = None
        return

    def __find_lanes(self, images):
        """Calibrates using chess images from camera_cal folder. Saves mtx and dist in TEST_FOLDER"""
        return images


    def test(self, img):
        test_images_paths = glob.glob('./test_images/*.jpg')
        test_images = []
        for fname in test_images_paths:
            test_images.append(mpimg.imread(fname))
            
        plt.imshow(__find_lane(test_images[0]))


    def run(self, img):
        return

if __name__ == '__main__':
    obj = Pipeline()
    obj.run()



