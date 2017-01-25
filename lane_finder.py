"""
Udacity Self Driving Car Nanodegree

Project 4 - Advanced Lane Finding
Advanced Lane Finding

---------
Tyler Lanigan
January, 2017

tylerlanigan@gmail.com
"""


import numpy as np
import cv2, pickle, glob, os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tools


from moviepy.editor import VideoFileClip
from IPython.display import HTML


# Stores objects and functions that are used for more than one frame.
# Stores controlling variables.
class GlobalObjects:

    def __init__(self):
        self.__set_folders()
        self.__set_hyper_parameters()
        self.__set_perspective()
        self.__set_mask_regions()
        self.__set_kernels()


    def __set_folders(self):
        # Use one slash for paths.
        self.camera_cal_folder = 'camera_cal/'
        self.test_images = glob.glob('test_images/*.jpg')
        self.output_image_path = 'output_images/test_'
        self.output_movie_path = 'output_movies/done_'


    def __set_hyper_parameters(self):
        self.img_size   = (640, 360) #(x,y) values for resized img
        return

    def __set_kernels(self):
        """Kernels used for image processing"""
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))


    def __set_perspective(self):
        # The src points draw a persepective trapezoid, the dst points draw
        # them as a square.  M transforms x,y from trapezoid to square for
        # a birds-eye view.  M_inv does the inverse.
        src = np.float32(((500, 548), (858, 548), (1138, 712), (312, 712)))*0.5
        dst = np.float32(((350, 600), (940, 600), (940, 720), (350, 720)))*0.5
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)
        self.dsize = (640,360)

    def __set_mask_regions(self):
        # We clip the bottom of the birds-eye view to eliminate reflections
        # from the car dashboard.  The roi_clip cuts a trapezoid from a normal
        # image.
        self.bottom_clip = np.int32(np.int32([[[60,0], [1179,0], [1179,650], [60,650]]])*0.5)
        self.roi_clip =  np.int32(np.int32([[[640, 425], [1179,550], [979,719],
                              [299,719], [100, 550], [640, 425]]])*0.5)



class LaneFinder(object):
    """
    The mighty LaneFinder takes in a video from the front camera of a self driving car
    and produces a new video with the traffic lanes highlighted and statistics about where
    the car is relative to the center of the lane shown.
    """    
    
    def __init__(self):

        self.g             = GlobalObjects()        
        self.thresholder   = tools.ImageThresholder()
        self.distCorrector = tools.DistortionCorrector(self.g.camera_cal_folder)

        return

    def __image_pipeline(self, img):
        """The pipeline for processing images. Globals g are added to functions that need
        access to global variables.
        """
        resized     = self.__resize_image(img, self.g)
        undistorted = self.__correct_distortion(resized)
        enhanced    = self.__enhance_image(undistorted, self.g)
        warped      = self.__warp_image_to_biv(enhanced)
        thresholded = self.__threshold_image(warped)
        bot_masked  = self.__mask_region(thresholded, self.g.bottom_clip)


      
        result = bot_masked
        
        return result

    def __mask_region(self, img, vertices):
        """Masks a region specified by clockwise vertices.
        """
        mask = np.zeros_like(img)   
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
        cv2.fillConvexPoly(mask, vertices, ignore_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image 

    def __enhance_image(self, img, g):
        """
        Enhances/sharpens the image using a clahe kernel
        See https://en.wikipedia.org/wiki/Adaptive_histogram_equalization
        """
        blue = self.g.clahe.apply(img[:,:,0])
        green = self.g.clahe.apply(img[:,:,1])
        red = self.g.clahe.apply(img[:,:,2])
        img[:,:,0] = blue
        img[:,:,1] = green
        img[:,:,2] = red
        return img

    def __resize_image(self, img, g):
        """
        Image is resized for memory purposes
        """
        return cv2.resize(img, self.g.img_size, 
                          interpolation = cv2.INTER_CUBIC)

    def __correct_distortion(self, img):
        return self.distCorrector.undistort(img)

    def __threshold_image(self, img):
        return self.thresholder.get_thresholded_image(img)

    def __warp_image_to_biv(self, img):
        return cv2.warpPerspective(img, self.g.M, self.g.img_size)

    def run(self, vid_input_path='project_video.mp4'):
        """
        Run code on the assigned project video.
        """
        vid_output_path = self.g.output_movie_path +  vid_input_path
        print('Finding lanes for ', vid_input_path)        
        self.__find_lanes( vid_input_path, vid_output_path)

        clip1 = VideoFileClip(vid_input_path)
        test_clip = clip1.fl_image(self.__image_pipeline)  
        test_clip.write_videofile(vid_output_path, audio=False)

        return True

    def test_one_image(self, img):
        """
        Tests the pipeline on one image
        """
        return self.__image_pipeline(img)

    def test(self, save=False):
        """
        Tests the __image_pipeline on all of the images
        in the testing folder.
        """
        print("Testing images...")

        # Save test images
        if save:
            for path in self.g.test_images:
                # Save Images
                image = (mpimg.imread(path))
                image = self.__image_pipeline(image)
                savep = self.g.output_image_path + path.split('/')[1]
                plt.imsave(savep, image)
            print('Test images saved.')

        # Display test images    
        else:

            fig = plt.figure(figsize=(10,12))
            i = 0            
            for path in self.g.test_images:
                #Display images
                ax = fig.add_subplot(4,2,i+1)
                img = mpimg.imread(path)
                img = self.__image_pipeline(img)
                plt.imshow(img, cmap='gray')
                plt.title(path.split('/')[1])
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                i += 1
            plt.tight_layout()
            plt.show()             

        return


if __name__ == '__main__':
    obj = LaneFinder()
    # obj.run()
    obj.test(save=False)



