
'''

'''
import imp as mp
import DrawLaneLines as dll
import Distortion as dstr
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import pickle
import pandas as pd
import DrawLaneLines as dll


def save_PERSP_to_pickle_file(M,Minv):
    ##to get image size
    dist_pickle = {}
    dist_pickle["M"] = M
    dist_pickle["Minv"] = Minv
    pickle.dump( dist_pickle, open( "./camera_cal/perspective_M.p", "wb" ))

def load_PERSP_from_pickle_file():
    ob = pd.read_pickle("./camera_cal/perspective_M.p")
    M=    ob["M"]
    Minv= ob["Minv"]
    return M,Minv                
                
def generate_perspective_pipeline():
    
    '''
    Function defines 4 source points and 4 destination points
    and returns M and Minv matrices
    
    We define source position by reusing lane lines detection used in the previous project
    FInal results are saved in "./camera_cal/perspective_M.p" pickle file
    
    '''

    image = mpimg.imread('test_images/straight_lines1.jpg')
    print('This image is:', type(image), 'with dimensions:', image.shape)
    #plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
    #plt.title('Original Image')
    #plt.show()
    
    
    mtx,dist=dstr.get_mtx_dist_from_pickle_file()
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    
    #plt.imshow(undist)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
    #plt.title('Undistorted')
    #plt.show()
    
    ##add pipeline

    gray=dll.grayscale(undist)
    #plt.imshow(gray, cmap='gray')
    #plt.title('Gray Scale')
    #plt.show()
    
    kernel_size = 7
    blur_gray = dll.gaussian_blur(gray, kernel_size)
    #plt.imshow(blur_gray, cmap='gray')
    #plt.title('Blured')
    #plt.show()
    
    low_threshold=85
    high_threshold=255

    edges = dll.canny(blur_gray, low_threshold, high_threshold)
    #plt.imshow(edges, cmap='gray')
    #plt.title('Gradient')
    #plt.show()
    
    vertices=dll.get_vertices(image)
    
    
    region=dll.region_of_interest(edges,np.array([vertices], dtype=np.int32))
    #plt.imshow(region,cmap='gray')
    #plt.title('Region Selection')
    #plt.show()
    
    hough_image,p1,p2,p1_hat,p2_hat=dll.hough_lines_trapezoid(region, 0.8, np.pi/180, 30, 50, 50)

    #plt.imshow(hough_image,cmap='gray')
    #plt.title('Lines Detection')
    #plt.show()
    
    
    final_image=dll.weighted_img(hough_image, image, α=0.8, β=1., γ=0.)
    #plt.imshow(final_image)
    #plt.title('Final Image')
    #plt.show()
    
    print('right line lower',p1)
    print('left line lower',p2)
    
    print('right line should be:',p1_hat)
    print('left line should be:',p2_hat)
    
    src =np.float32([[p1[0],p1[1]],[p1[2],p1[3]],[p2[0],p2[1]],[p2[2],p2[3]]])
    dst =np.float32([[1000,0],[1000,700],[200,0],[200,700]])
    
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    save_PERSP_to_pickle_file(M,Minv)
    
    
    f, (ax1, ax2,ax3,ax4,ax5,ax6,ax7) = plt.subplots(1, 7, figsize=(20,10))
    ax1.imshow(image,cmap='gray')
    ax1.set_title('original image', fontsize=20)
    ax2.imshow(gray ,cmap='gray')
    ax2.set_title('gray', fontsize=20)
    ax3.imshow(blur_gray,cmap='gray')
    ax3.set_title('blur_gray', fontsize=20)
    ax4.imshow(edges,cmap='gray')
    ax4.set_title('edges', fontsize=20)
    ax5.imshow(region,cmap='gray')
    ax5.set_title('region ', fontsize=20)
    ax6.imshow(hough_image,cmap='gray')
    ax6.set_title('hough_image ', fontsize=20)
    ax7.imshow(final_image,cmap='gray')
    ax7.set_title('hough_image ', fontsize=20)
 
                
def get_unwarped_image(image):

    mtx,dist=dstr.get_mtx_dist_from_pickle_file()
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    
    print('Image shape',undist.shape)
    M,Minv=load_PERSP_from_pickle_file()
    warped = cv2.warpPerspective(undist, M, (undist.shape[1],undist.shape[0]))
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(warped)
    ax2.set_title('Warped Image', fontsize=30)      
                
                
                
                
                
                