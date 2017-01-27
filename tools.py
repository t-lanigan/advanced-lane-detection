
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import os
import glob
from scipy import signal


class Line:
    """
    The line class defines a bunch of characteristics of a single line (lane line)
    It also includes a function to return the curvature of the line.
    """
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

        #Conversions from pixels to real measurements
        self.ym_per_pix = 30/720
        self.xm_per_pix = 3.7/700

    def get_curvature(self, which_fit='best'):
        """
        Returns the curvature of the line.
        """
        
        if which_fit == 'best':
            fit = self.best_fit
        else:
            fit = self.current_fit

        y_eval = np.max(self.ally)

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meteres per pixel in x dimension

        fit_cr = np.polyfit(self.ally*self.ym_per_pix, 
                            self.allx*self.xm_per_pix, 2)
        
        #Radius of curvature formula.
        self.radius_of_curvature = ((1 + (2*fit_cr[0]*y_eval + fit_cr[1])**2)**1.5) \
                                     /np.absolute(2*fit_cr[0])
        return self.radius_of_curvature


class HistogramLineFitter:
    """
    The HistogramLineFitter uses an adaptive histogram fitting technique to 
    determine where the lanes most likely are.

    A line is defined as a yellow or white single line in traffic. A lane is a

    combinations of two of the lines.
    """

    def __init__(self):

        return

    def get_line(self, img, line, direction="left"):

        # Window dimensions for histograms sliding window.
        # see _get_histogram method for ye,ys,xs, and xe explainations
        
        win_width = 25
        win_height = 50 

        if not line.detected:

            xm = img.shape[1]
            ym = img.shape[0]
            h = self.__get_histogram(img, ym*(.5), ym, 0, xm)


            # Find both peaks
            peaks = signal.find_peaks_cwt(h, np.arange(100,200))
            if direction == 'left':
                peak = peaks[0]
            else:
                peak = peaks[-1]
            
            # Move the sliding window and gather the associated points.
            yvals = []
            xvals = []
            
            for i in range(win_height):

                if direction == 'left':
                    if peak < win_width:
                        peak = win_width
                else:
                    if peak >= (xm - win_width):
                        peak = xm - win_width - 1
                
                start_range = int(ym*((win_height-i-1) / win_height))
                end_range = int(ym*((win_height-i) / win_height))

                for yval in range(start_range , end_range):
                    for xval in range(peak-win_width, peak + win_width):
                        if img[yval][xval] == 1.0:
                            yvals.append(yval)
                            xvals.append(xval)
                # Find new peaks to move the window for next iteration
                # new peaks will be the max in the current window plus 
                # the beginning of the window...
                
                ## See __get_histogram function for explaination.
                ye = ym *((win_height-i-1)/win_height) 
                ys = ym *((win_height-i)/win_height)
                xs = peak-win_width
                xe = peak+win_width

                h = self.__get_histogram(img, ye, ys, xs, xe)
                if len(signal.find_peaks_cwt(h, np.arange(100,200))) > 0:
                    peak = np.amax(signal.find_peaks_cwt(h, np.arange(100,200))) + xs
                
                else: 
                # Look in bigger window
                    win_width_big = 100
                    ye = ym*((win_height-i-1)/win_height)
                    ys = ym*((win_height-i)/win_height)
                    xs = peak-win_width_big
                    xe = peak+win_width_big

                    h = self.__get_histogram(img, ye, ys, xs, xe)

                    if len(h > 0):
                        if len(signal.find_peaks_cwt(h, np.arange(100,200))) > 0:
                            peak = np.amax(signal.find_peaks_cwt(h, np.arange(100,200))) + xs

            yvals = np.asarray(yvals)
            xvals = np.asarray(xvals)
           
            line.allx = xvals
            line.ally = yvals
            
            # Fit a second order polynomial to lane line
            fit = np.polyfit(yvals, xvals, 2)
            
            line.current_fit = fit
            line.best_fit = fit

            
            fitx = fit[0]*yvals**2 + fit[1]*yvals + fit[2]

            
            line.recent_xfitted.append(fitx)
            line.bestx = fitx
            
        else:
            #initial peak - use previous line x
            peak = line.bestx[0]
            prev_line = copy(line)
            
            #move the sliding window across and gather the points
            yvals = []
            xvals = []
            
            for i in range(win_height):
                #peaks may be at the edge so we need to stop at the edge
                if direction == 'left':
                    if int(peak) < win_width:
                        peak = win_width
                else:
                    if int(peak) >= (xm - win_width):
                        peak = xm - win_width - 1
                        
                start_range = int(ym*((win_height-i-1)/win_height))
                end_range = int(xm*((win_height-i)/win_height))

                for yval in range(start_range, end_range):
                    for xval in range(int(peak-win_width), int(peak+win_width)):
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
            
            is_ok = self.__check_detection(prev_line, line)
            if is_ok:
                if len(line.recent_xfitted) > 10:
                    #remove the first element
                    line.recent_xfitted.pop(0)
                    line.recent_xfitted.append(fitx)
                    line.bestx = fitx
                    line.best_fit = fit
            else:
                # Line lost, go back to sliding window
                line.detected = false
            
        return line


    def __get_histogram(self, img, y_end, y_start, x_start, x_end):
        """
        Returns a histogram in the given windows. The images have the y axis pointing down
        of the z-axis pointing into the screen. The y_end, and x_end are the larger pixel limits
        of the window of the histogram.
                        |
        y_start-->  120 |
                        |
        y_end-->   360  |
                        |___________________________________________
                                  ^               ^
                               x_start (200)      x_end (400)
        """

        return np.sum(img[y_end:y_start , x_start:x_end], axis=0)

    def __check_detection(self, prev_line, next_line):
        """
        Checks two lines to see if they have similar curvature.
        """

        left_c = prev_line.get_curvature(which_fit='current')
        right_c = next_line.get_curvature(which_fit='current')
        # Checking that they are separated by approximately the right distance horizontally
        left_x = prev_line.recent_xfitted[0][0]
        right_x = next_line.recent_xfitted[0][0]
        if (np.absolute(left_x - right_x) > 1000) | (np.absolute(left_c - right_c) > 100):
            prev_line.detected = False
            next_line.detected = False
            return False

        prev_line.detected = True #in case these are different lines that are being compared
        next_line.detected = True
        return True


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
        pts_left = np.array([np.transpose(np.vstack([lines['left_line'].allx, 
                                                    lines['left_line'].ally]))])

        pts_right = np.array([np.flipud(np.transpose(np.vstack([lines['right_line'].allx, 
                                                               lines['right_line'].ally])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        
        color_warp = self.__draw_lane_pixels(lines['left_line'], color_warp)
        color_warp = self.__draw_lane_pixels(lines['right_line'], color_warp)

        #get the radius curvature
        left_curverad = lines['left_line'].get_curvature(which_fit='best')
        right_curverad = lines['right_line'].get_curvature(which_fit='best')

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

    def __draw_lane_pixels(self, line, img):
        """
        Draws the pixels associated with the allx and ally coordinates in the line.

        Change the colour with the tuplet.
        """
        for idx,pt in enumerate(line.ally):
            cv2.circle(img,(line.allx[idx], pt), 2, (255,0,0), -1)

        return img

        
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




