import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import os
import glob



class LaneHistogramFitter:
    """Takes in a bitmap image of two lanes and  uses an adaptive histogram gitting method to 
    return the x, y coordinates of each of the lanes.
    """
    def __init__(self, g):

        self.g = g

    def histogram_fit(self, img):

        # Use intensity histograms in the x (vertical) direction to detect
        # potential lane markers on the left and ride side of the image.
        # Feed these to our makers who will figure out if they're good and
        # compute a moving average, which we'll use for guidance.
        #

def myPipeline(img):
    ksize = 3
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(10, 100))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(5, 250))
    mag_binary = mag_threshold(img, sobel_kernel=ksize, mag_thresh=(5, 100))
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(0, np.pi/2))
    s_binary = color_threshold_hsv(img, "s", (120,255))
    v_binary = color_threshold_yuv(img,"v", (0,105))
    r_binary = color_threshold_rgb(img,"r", (230,255))
    result = np.zeros_like(dir_binary)
    result[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) & ((s_binary == 1)) | ((v_binary ==1) | (r_binary == 1))] = 1
    return result





# Function that takes image, kernel size, and threshold and returns
# magnitude of the gradient
def mag_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

#function to threshold HSV color spectrum in an image for a given range
def color_threshold_hsv(img, channel="s", thresh=(170,255)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hsv[:,:,0]
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    
    if channel == "h":
        target_channel = h_channel
    elif channel == "l":
        target_channel = l_channel
    else:
        target_channel = s_channel
    
    # Threshold color channel
    binary_output = np.zeros_like(target_channel)
    binary_output[(target_channel >= thresh[0]) & (target_channel <= thresh[1])] = 1
    
    return binary_output

#function to threshold RGB color spectrum in an image for a given range
def color_threshold_rgb(img, channel="r", thresh=(170,255)):
    img = np.copy(img)
    r_channel = img[:,:,0]
    g_channel = img[:,:,1]
    b_channel = img[:,:,2]
    
    if channel == "r":
        target_channel = r_channel
    elif channel == "g":
        target_channel = g_channel
    else:
        target_channel = b_channel
    
    # Threshold color channel
    binary_output = np.zeros_like(target_channel)
    binary_output[(target_channel >= thresh[0]) & (target_channel <= thresh[1])] = 1
    
    return binary_output
    
#function to threshold HSV color spectrum in an image for a given range
def color_threshold_yuv(img, channel="v", thresh=(0,255)):
    img = np.copy(img)
    # Convert to YUV color space and separate the V channel
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV).astype(np.float)
    y_channel = yuv[:,:,0]
    u_channel = yuv[:,:,1]
    v_channel = yuv[:,:,2]
    
    if channel == "y":
        target_channel = y_channel
    elif channel == "u":
        target_channel = u_channel
    else:
        target_channel = v_channel
    
    # Threshold color channel
    binary_output = np.zeros_like(target_channel)
    binary_output[(target_channel >= thresh[0]) & (target_channel <= thresh[1])] = 1
    
    return binary_output

# Function to threshold gradient direction in an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output



 
class ImageThresholder:
    """
    The ImageThresholder takes in an rgb image and spits out a thresholded image 
    using a variety of techniques. Filtering techniques aim to extract the yellow and 
    white traffic lines for a variety of conditions.
    """

    def __init__(self):
        return

    def __generate_colors_spaces(self):
        self.hsv = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2HSV)
        self.yuv = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2YUV)
        self.gray = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)

    def get_thresholded_image(self, rgb):
        self.rgb = rgb
        self.thresh = np.zeros((self.rgb.shape[0], self.rgb.shape[1]), dtype=np.uint8)
        self.__generate_colors_spaces()
        self.__add_yellow_pixels()
        self.__add_white_pixels()
        self.__add_sobel_thresholds()
        self.__ignore_shadows()
        return self.thresh

    def __add_yellow_pixels(self):
        """Yellow pixels are identified through a band-pass filter in the HSV
        (hue, saturation, value/brightness) space.
        """
        lower  = np.array([ 0, 80, 200])
        upper = np.array([ 40, 255, 255])
        yellows = np.array(cv2.inRange(self.hsv, lower, upper))
        self.thresh[yellows > 0] = 1

    def __add_white_pixels(self):
        """White filtering method from "Real-Time Lane Detection and Rear-End 
        Collision Warning SystemOn A Mobile Computing Platform", Tang et.al., 2015
        """
        y = self.yuv[:,:,0]
        whites = np.zeros_like(y)
        bits = np.where(y  > 100)  # was 175
        whites[bits] = 1

        # Define a filter kernel to find the white pixels.
        kernel = np.ones((11,11),np.float32)/(1-11*11)
        kernel[5,5] = 1.0
        mask2 = cv2.filter2D(y,-1,kernel)
        whites[mask2 < 5] = 0
        self.thresh = self.thresh | whites 

    def __add_sobel_thresholds(self):
        """The green channel in an rgb image, and the gray image are 
        good candidates for finding edges. As talked about in class,
        and by experimentaion.
        """
        green = self.__abs_sobel_thresh(self.rgb[:,:,1])
        shadows = self.__abs_sobel_thresh(self.gray, thresh_min=10, thresh_max=64)
        self.thresh = self.thresh | green | shadows

    def __abs_sobel_thresh(self, gray, orient='x', thresh_min=20, thresh_max=100):
        """Apply a Sobel filter to find edges, scale the results
        from 1-255 (0-100%), then use a band-pass filter to create a mask
        for values in the range [thresh_min, thresh_max].
        """
        sobel = cv2.Sobel(gray, cv2.CV_64F, (orient=='x'), (orient=='y'))
        abs_sobel = np.absolute(sobel)
        max_sobel = max(1,np.max(abs_sobel))
        scaled_sobel = np.uint8(255*abs_sobel/max_sobel)
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        return binary_output

    def __ignore_shadows(self):
        """Find brighter spots on the road and ignore the really dark areas.
        """
        bits = np.zeros_like(self.gray)
        thresh = np.mean(self.gray)
        bits[self.gray > thresh] = 1
        self.thresh = self.thresh & bits



class DistortionCorrector:    
    """Takes in path to calibration files as string.

    Params:
    self.nx = How many inside corners in chess boards (y-direction)
    self.ny = How many inside corner in chess boards (x-direction)
    calibration_folder_path = path to calibration images.

    Methods:

    fit: Calibrates the distorsionCorrector with array of images [None, width, height, channels]
    undistort: takes in an image, and outputs undistorted image
    test: takes in an image, and displays undistored image alongside original.

    -----------
    In this project it is already fitted, however it can be used for other projects.

    To Fit:

    # cal_images_paths = glob.glob('./camera_cal/cal*.jpg')
    # cal_images = []
    # for fname in cal_images_paths:
    #     cal_images.append(mpimg.imread(fname))
    # distCorrector.fit(cal_images)

    """  
    def __init__(self, calibration_folder_path):

        # Set nx and ny according to how many inside corners in chess boards images.  
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
            print('Mtx and dist matrix missing. Please call fit distortionCorrector')
        return

    def fit(self, images):
        """Calibrates using chess images from camera_cal folder. 
        Saves mtx and dist in calibration_folder_path
        """
        
        cname = self.cal_folder + 'calibration.p'
        if  os.path.isfile(cname):
            print('Deleting existing calibration files...')
            os.remove(cname)

        print("Computing camera calibration...")

        objp = np.zeros((self.ny*self.nx,3), np.float32)
        objp[:,:2] = np.mgrid[0:self.nx,0:self.ny].T.reshape(-1,2)

        objpoints = []
        imgpoints = [] 


        # Step through the list and search for chessboard corners
        for img in images:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (self.nx,self.ny), None)
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




