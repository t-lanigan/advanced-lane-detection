import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import os
import glob
from scipy import signal

# From Udacity: Define a class to receive the characteristics of each line detection
class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None


class HistogramLineFitter:

    def __init__(self):

        return

    def get_line(self, img, line, direction="left"):

        #bigger numbers create smaller windows as it is calculated as a proportion to the img height
        winWidth = 25
        winHeight = 50 

        if not line.detected:
            histogram = np.sum(img[img.shape[0]*(.5):,0:img.shape[1]], axis=0)
            #plt.plot(histogram)

            #find two peaks first is the left and last is the right
            #initial peak
            peakind = signal.find_peaks_cwt(histogram, np.arange(100,200))
            if direction == 'left':
                peak = peakind[0]
            else:
                peak = peakind[-1]
            
            #move the sliding window across and gather the points
            yvals = []
            xvals = []
            
            for i in range(winHeight):
                #peaks may be at the edge so we need to stop at the edge
                if direction == 'left':
                    if peak < winWidth:
                        peak = winWidth
                else:
                    if peak >= (img.shape[1] - winWidth):
                        peak = img.shape[1] - winWidth - 1
                
                for yval in range(int(img.shape[0]*((winHeight-i-1)/winHeight)), int(img.shape[0]*((winHeight-i)/winHeight))):
                    for xval in range(peak-winWidth, peak+winWidth):
                        if img[yval][xval] == 1.0:
                            yvals.append(yval)
                            xvals.append(xval)
                #find new peaks to move the window accordingly for next iteration
                #new peaks will be the max in the current window plus the beginning of the window...
                
                histogram = np.sum(img[img.shape[0]*((winHeight-i-1)/winHeight):img.shape[0]*((winHeight-i)/winHeight),peak-winWidth:peak+winWidth], axis=0)
                if len(signal.find_peaks_cwt(histogram, np.arange(100,200))) > 0:
                    peak = np.amax(signal.find_peaks_cwt(histogram, np.arange(100,200))) + (peak-winWidth)
                else: #look in bigger window
                    winWidthBig = 100
                    histogram = np.sum(img[img.shape[0]*((winHeight-i-1)/winHeight):img.shape[0]*((winHeight-i)/winHeight),peak-winWidthBig:peak+winWidthBig], axis=0)
                    if len(histogram > 0):
                        if len(signal.find_peaks_cwt(histogram, np.arange(100,200))) > 0:
                            peak = np.amax(signal.find_peaks_cwt(histogram, np.arange(100,200))) + (peak-winWidthBig)

            yvals = np.asarray(yvals)
            xvals = np.asarray(xvals)
           
            line.allx = xvals
            line.ally = yvals
            
            # Fit a second order polynomial to lane line
            fit = np.polyfit(yvals, xvals, 2)
            
            line.current_fit = fit
            line.best_fit = fit
            #print(fit)
            
            fitx = fit[0]*yvals**2 + fit[1]*yvals + fit[2]
            #print(fitx)
            
            line.recent_xfitted.append(fitx)
            line.bestx = fitx
            
        else:
            #initial peak - use previous line x
            peak = line.bestx[0]
            prev_line = copy(line)
            
            #move the sliding window across and gather the points
            yvals = []
            xvals = []
            
            for i in range(winHeight):
                #peaks may be at the edge so we need to stop at the edge
                if direction == 'left':
                    if int(peak) < winWidth:
                        peak = winWidth
                else:
                    if int(peak) >= (img.shape[1] - winWidth):
                        peak = img.shape[1] - winWidth - 1
                        
                for yval in range(int(img.shape[0]*((winHeight-i-1)/winHeight)), int(img.shape[0]*((winHeight-i)/winHeight))):
                    for xval in range(int(peak-winWidth), int(peak+winWidth)):
                        if img[yval][xval] == 1.0:
                            yvals.append(yval)
                            xvals.append(xval)
                #use bestx to keep going over the line
                peak = line.bestx[(i + 1)%len(line.bestx)]

            yvals = np.asarray(yvals)
            xvals = np.asarray(xvals)
            
            line.allx = xvals
            line.ally = yvals
            
            # Fit a second order polynomial to lane line
            fit = np.polyfit(yvals, xvals, 2)
            line.current_fit = fit
            fitx = fit[0]*yvals**2 + fit[1]*yvals + fit[2]
            
            isOk = self.__check_detection(prev_line, line)
            if isOk:
                if len(line.recent_xfitted) > 10:
                    #remove the first element
                    line.recent_xfitted.pop(0)
                    line.recent_xfitted.append(fitx)
                    line.bestx = fitx
                    line.best_fit = fit
            else:
                #line lost, go back to sliding window
                line.detected = false
            
        return line


class LaneDrawer:
    """
    Takes in lane lines, a warped image and an undistorted image and draws the lanes with a 
    cv2.fillPoly
    """ 
    def __init__(self):
        return

    def draw_lanes(self, undist, warped, lines, Minv):

        undist = np.copy(undist)
        img = np.copy(warped)
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([lines['left_line'].allx, lines['left_line'].ally]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([lines['right_line'].allx, lines['right_line'].ally])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        
        #Draw the points on the image
        for idx,pt in enumerate(lines['left_line'].ally):
            #cv2.circle(img,(447,63), 63, (0,0,255), -1)
            cv2.circle(color_warp,(lines['left_line'].allx[idx], pt), 2, (255,0,0), -1)
        
        for idx,pt in enumerate(lines['right_line'].ally):
            #cv2.circle(img,(447,63), 63, (0,0,255), -1)
            cv2.circle(color_warp,(lines['right_line'].allx[idx], pt), 2, (0,0,255), -1)
        
        #get the radius curvature
        left_curverad = self.__get_curvature(lines['left_line'].allx, lines['left_line'].ally, lines['left_line'].best_fit)
        right_curverad = self.__get_curvature(lines['right_line'].allx, lines['right_line'].ally, lines['right_line'].best_fit)
        left_text = 'Left Curvature Radius: ' + str(np.around(left_curverad,2)) + 'm'
        right_text = 'Right Curvature Radius: ' + str(np.around(right_curverad,2)) + 'm'
        
        #get the distance from the center
        center_diff = self.__get_center_difference(undist, lines)
        if center_diff < 0:
            center_diff_text = 'Vehicle Position: ' + str(np.around(np.absolute(center_diff),2)) + 'm left of center'
        else:
            center_diff_text = 'Vehicle Position: ' + str(np.around(center_diff,2)) + 'm right of center'
            
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(undist,left_text,(10,50), font, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(undist,right_text,(10,100), font, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(undist,center_diff_text,(10,150), font, 1,(255,255,255),2,cv2.LINE_AA)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        
        return result

    def __check_detection(self, prev_line, next_line):
        # Checking that they have similar curvature
        left_curvature = getCurvature(prev_line.allx, prev_line.ally, prev_line.current_fit )
        right_curvature = getCurvature(next_line.allx, next_line.ally, next_line.current_fit)
        # Checking that they are separated by approximately the right distance horizontally
        left_x = prev_line.recent_xfitted[0][0]
        right_x = next_line.recent_xfitted[0][0]
        if (np.absolute(left_x - right_x) > 1000) | (np.absolute(left_curvature - right_curvature) > 100): #in pixels, not meters
            prev_line.detected = False
            next_line.detected = False
            return False

        prev_line.detected = True #in case these are different lines that are being compared
        next_line.detected = True
        return True

        # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    def __get_curvature(self, line_x, line_y, fit):
        
        y_eval = np.max(line_y)
        curverad = ((1 + (2*fit[0]*y_eval + fit[1])**2)**1.5) \
                                     /np.absolute(2*fit[0])
        
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meteres per pixel in x dimension

        fit_cr = np.polyfit(line_y*ym_per_pix, line_x*xm_per_pix, 2)
        
        curverad = ((1 + (2*fit_cr[0]*y_eval + fit_cr[1])**2)**1.5) \
                                     /np.absolute(2*fit_cr[0])
        
        # Now our radius of curvature is in meters
        #print(left_curverad, 'm', right_curverad, 'm')
        # Example values: 3380.7 m    3189.3 m

        return curverad
        # Example values: 1163.9    1213.7
        
    def __get_center_difference(self, img,lines):
        #midpoint of the lines (half the polyfill width)
        midPoly = (lines['right_line'].bestx[0] - lines['left_line'].bestx[0]) / 2
        #midpoint of the image (half the image length)
        midImage = img.shape[0] / 2
        
        diffInPix = midImage - midPoly
        #convert to meters
        xm_per_pix = 3.7/700 # meteres per pixel in x dimension
        result = diffInPix * xm_per_pix
        lines['left_line'].line_base_pos = result
        lines['right_line'].line_base_pos = result
        return result
        
 
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
        self.__generate_colors_spaces()
        gradx = self.__abs_sobel_thresh(orient='x', thresh=(10, 100))
        grady = self.__abs_sobel_thresh(orient='y', thresh=(5, 250))
        mag_binary = self.__mag_threshold(mag_thresh=(5, 100))
        dir_binary = self.__dir_threshold(dir_thresh=(0, np.pi/2))
        s_binary = self.__color_threshold_hsv("s", (120,255))
        v_binary = self.__color_threshold_yuv("v", (0,105))
        r_binary = self.__color_threshold_rgb("r", (230,255))
        self.thresh = np.zeros_like(dir_binary)

        #Combine results
        self.thresh[((gradx == 1) & (grady == 1)) | ((mag_binary == 1)
               & (dir_binary == 1)) & ((s_binary == 1))
               | ((v_binary ==1) | (r_binary == 1))] = 1

        return self.thresh

    def __color_threshold_hsv(self, channel="s", thresh=(170,255)):
        """Band pass filter for HSV colour space"""

        h, s, v = cv2.split(self.hsv)

        if channel == "h":
            target_channel = h
        elif channel == "l":
            target_channel = s
        else:
            target_channel = v

        binary_output = np.zeros_like(target_channel)
        binary_output[(target_channel >= thresh[0]) & (target_channel <= thresh[1])] = 1
        
        return binary_output


    def __color_threshold_rgb(self, channel="r", thresh=(170,255)):
        """Band pass filter for RGB colour space"""

        r,g,b = cv2.split(self.rgb)
        
        if channel == "r":
            target_channel = r
        elif channel == "g":
            target_channel = g
        else:
            target_channel = b

        binary_output = np.zeros_like(target_channel)
        binary_output[(target_channel >= thresh[0]) & (target_channel <= thresh[1])] = 1
        
        return binary_output

    def __color_threshold_yuv(self, channel="v", thresh=(0,255)):
        """Band pass filter for YUV colour space"""

        y, u, v  = cv2.split(self.yuv)
        
        if channel == "y":
            target_channel = y
        elif channel == "u":
            target_channel = u
        else:
            target_channel = v

        binary_output = np.zeros_like(target_channel)
        binary_output[(target_channel >= thresh[0]) & (target_channel <= thresh[1])] = 1
        
        return binary_output


    def __abs_sobel_thresh(self, orient='x', thresh=(0,255)):
        """Apply a Sobel filter to find edges, scale the results
        from 1-255 (0-100%), then use a band-pass filter to create a mask
        for values in the range [thresh_min, thresh_max].
        """
        sobel = cv2.Sobel(self.gray, cv2.CV_64F, (orient=='x'), (orient=='y'))
        abs_sobel = np.absolute(sobel)
        max_sobel = max(1,np.max(abs_sobel))
        scaled_sobel = np.uint8(255*abs_sobel/max_sobel)
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return binary_output


    def __mag_threshold(self, sobel_kernel=3, mag_thresh=(0, 255)):
        """
        Function that takes image, kernel size, and threshold and returns
        magnitude of the gradient
        """

        sobelx = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        scale_factor = np.max(gradmag)/255
        gradmag = (gradmag/scale_factor).astype(np.uint8)
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        return binary_output


    def __dir_threshold(self, sobel_kernel=3, dir_thresh=(0, np.pi/2)):
        """
        Function to threshold gradient direction in an image for a given 
        range and Sobel kernel.
        """

        sobelx = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= dir_thresh[0]) & (absgraddir <= dir_thresh[1])] = 1

        return binary_output




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




