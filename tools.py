import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import os
import glob


class Thresholder(object):

    def __init__(self):

        """
        threshold_img takes in an image and returns a single channel 
        binary image detection the edges. 

        """
        self.s_thresh = (170, 255)
        self.sx_thresh = (20, 100)


    def threshold_img(self, img):
        
        hls = np.copy(img)
        hls = self.__get_hls_transform(img)
        s_channel = self.__get_s_channel(hls)


        scaled_sobel = self.__get_scaled_sobelx(s_channel)

        # Threshold x gradient
        thresh_min = self.sx_thresh[0]
        thresh_max = self.sx_thresh[1]
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        # Threshold color channel
        s_thresh_min = self.s_thresh[0]
        s_thresh_max = self.s_thresh[1]
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

        return combined_binary

    def __get_scaled_sobelx(self,img):
        """Only takes in one channel."""

        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0) # Take the derivative in x
        
         # Absolute x derivative to accentuate lines away from horizontal
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        return scaled_sobel

    def __get_hls_transform(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    def __get_s_channel(self, hls):
        return hls[:,:,2]


class distortionCorrector(object):    
    
    def __init__(self, calibration_folder_path):
        """Takes in path to calibration files as string.

        Params:
        self.nx = How many inside corners in chess boards (y-direction)
        self.ny = How many inside corner in chess boards (x-direction)
        calibration_folder_path = path to calibration images.

        Methods:

        fit: Calibrates the distorsionCorrector with array of images [None, width, height, channels]
        undistort: takes in an image, and outputs undistorted image
        test: takes in an image, and displays undistored image alongside original.
        """

        # Set nx and ny according to how many inside corners in chess boards.
        self.nx = 9
        self.ny = 6
        self.mtx = []
        self.dist = []
        self.cal_folder = calibration_folder_path

        fname = self.cal_folder + 'calibration.p'

        if  os.path.isfile(fname):
            print('Loading saved calibration file...')
            self.mtx, self.dist = pickle.load( open( fname, "rb" ) )
        else:
            print('Mtx and dist matrix missing. Please call .fit function.')
        return

    def fit(self, images):
        """Calibrates using chess images from camera_cal folder. Saves mtx and dist in TEST_FOLDER"""
        
        cname = self.cal_folder + 'calibration.p'
        if  os.path.isfile(cname):
            print('Deleting existing calibration files...')
            os.remove(cname)

        print("Computing camera calibration...")


        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.ny*self.nx,3), np.float32)
        objp[:,:2] = np.mgrid[0:self.nx,0:self.ny].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = []
        imgpoints = [] 


        # Step through the list and search for chessboard corners
        for img in images:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (self.nx,self.ny), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

        if not ret:
            raise ValueError('Most likely the self.nx and self.ny are not set correctly')

        img = images[0]


        # Calibrate the camera and get mtx, and dist matricies.
        _, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objpoints,
                                                 imgpoints,
                                                 img.shape[:-1],
                                                 None, None)

        pname = self.cal_folder + 'calibration.p'
        print("Pickling calibration files..")
        pickle.dump( (self.mtx, self.dist), open( pname, "wb" ) )

        return


    def undistort(self, img):
        """Returns undistored image"""

        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)


    def test(self, img):

        undist = self.undistort(img)
        
        f, (ax1, ax2) = plt.subplots(1, 2)
        f.tight_layout()
        ax1.imshow(img)
        ax1.set_title('Original Image')
        ax2.imshow(undist)
        ax2.set_title('Undistorted Image')

        plt.show()
        return




