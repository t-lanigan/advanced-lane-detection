import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tools
import pickle
import os
import glob

from moviepy.editor import VideoFileClip
from IPython.display import HTML





class LaneFinder(object):    
    
    def __init__(self):
        
        self.distCorrector = tools.distortionCorrector('./camera_cal/')

        
        # Already fitted, so I'll comment this out  
        # cal_images_paths = glob.glob('./camera_cal/cal*.jpg')
        # cal_images = []
        # for fname in cal_images_paths:
        #     cal_images.append(mpimg.imread(fname))
        # distCorrector.fit(cal_images)

        self.thresholder = tools.Thresholder()

        return

    def __image_pipeline(self, img):
        
        # Undistort the image.
        undistort = self.distCorrector.undistort(img)
        
        # Apply tuned binary thresholding.
        thresh = self.thresholder.threshold_img(undistort)


        return thresh
               
    def __find_lanes(self, input_path, output_path):              
        clip1 = VideoFileClip(input_path)
        test_clip = clip1.fl_image(self.__image_pipeline)  
        test_clip.write_videofile(output_path, audio=False)

    def run(self, vid_input='project_video.mp4'):
        """
        Run code on the assigned project video.
        """
        vid_output_path = './output_movies/completed_'+  vid_input
        print('Finding lanes for ', vid_input)
        
        
        self.__find_lanes( vid_input, vid_output_path)
        return True

    def test(self, save=False):
        """
        Tests the __image_pipeline on all of the images.
        """

        # Read in test image paths.
        img_paths = glob.glob('./test_images/*.jpg')

        if save:
            for path in img_paths:
                # Save Images
                image = (mpimg.imread(path))
                image = self.__image_pipeline(image)
                savep = 'output_images/test_'+ path.split('_images/')[1]
                plt.imsave(savep, image)             

        else:
            # Display Test Images
            fig = plt.figure(figsize=(10,12))
            i = 0            
            for path in img_paths:
                #Display images
                ax = fig.add_subplot(4,2,i+1)
                img = mpimg.imread(path)
                img = self.__image_pipeline(img)
                plt.imshow(img)
                plt.title(path.split('images/')[1])
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                i += 1
            plt.tight_layout()
            plt.show()             

        return


if __name__ == '__main__':
    obj = LaneFinder()
    #obj.run()
    obj.test()



