import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import os
import glob

TEST_FOLDER = './camera_cal/'
CAL_IMAGES = TEST_FOLDER + 'calibration*.jpg'
TEST_IMAGE = 'test-cal.jpg'


 
class distortionCorrector(object):    
    
    def __init__(self):

        # Set these according to how many sin
        self.nx = 9
        self.ny = 6
        self.mtx = []
        self.dist = []

        fname = TEST_FOLDER + 'calibration.p'

        if  os.path.isfile(fname):
            print('Loading saved calibration file.')
            self.mtx, self.dist = pickle.load( open( fname, "rb" ) )
        else:
            self.__calibrate()
        return

    def __calibrate(self):
        """Calibrates using chess images from camera_cal folder. Saves mtx and dist in TEST_FOLDER"""
        
        print("Computing camera calibration.")


        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.ny*self.nx,3), np.float32)
        objp[:,:2] = np.mgrid[0:self.nx,0:self.ny].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = []
        imgpoints = [] 

        # Make a list of calibration images
        images = glob.glob(CAL_IMAGES)

        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (self.nx,self.ny), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

        if not ret:
            raise ValueError('Most likely the self.nx and self.ny are not set correctly')

        fname = TEST_FOLDER + TEST_IMAGE
        img = cv2.imread(fname)


        # Calibrate the camera and get mtx, and dist matricies.
        _, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objpoints,
                                                 imgpoints,
                                                 img.shape[:-1],
                                                 None, None)

        pname = TEST_FOLDER + 'calibration.p'
        print("Pickling calibration files..")
        pickle.dump( (self.mtx, self.dist), open( pname, "wb" ) )

        return


    def undistort(self, img):
        """Returns undistored image"""

        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def run(self):
#       fname = './test_images/test1.jpg'
        fname = TEST_FOLDER + TEST_IMAGE
        img = cv2.imread(fname)
        img = img[...,::-1] #convert from opencv bgr to standard rgb
        
        undist = self.undistort(img)
        
        f, (ax1, ax2) = plt.subplots(1, 2)
        f.tight_layout()
        ax1.imshow(img)
        ax1.set_title('Original Image')
        ax2.imshow(undist)
        ax2.set_title('Undistorted Image')

        plt.show()
        return

if __name__ == '__main__':
    obj = distortionCorrector()
    obj.run()



