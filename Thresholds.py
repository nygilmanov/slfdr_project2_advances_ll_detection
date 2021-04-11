

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


def hist(img):
    # TO-DO: Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[img.shape[0]//2:,:]

    # TO-DO: Sum across image pixels vertically - make sure to set `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half, axis=0)
    plt.plot(histogram)
    plt.title('Define peaks on the histogram')
    
def get_warped_image(image):

    mtx,dist=get_mtx_dist_from_pickle_file()
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    
    print('Image shape',undist.shape)
    M,Minv=load_PERSP_from_pickle_file()
    warped = cv2.warpPerspective(undist, M, (undist.shape[1],undist.shape[0]))
    
    return warped 

def sobel_x_transform(img):
    # Transform image to gray scale
    gray_img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply sobel (derivative) in x direction, this is usefull to detect lines that tend to be vertical
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)

    # Scale result to 0-255
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    sx_binary = np.zeros_like(scaled_sobel)
    # Keep only derivative values that are in the margin of interest
    sx_binary[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1
    return sx_binary  

def get_white(img):
    
    gray_img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect pixels that are white in the grayscale image
    white_binary = np.zeros_like(gray_img)
    white_binary[(gray_img > 200) & (gray_img <= 255)] = 1
    return white_binary

def get_saturation(img):
    # Convert image to HLS
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    H = hls[:,:,0]
    S = hls[:,:,2]
    sat_binary = np.zeros_like(S)

    # Detect pixels that have a high saturation value
    sat_binary[(S > 90) & (S <= 255)] = 1
    
    hue_binary =  np.zeros_like(H)
    # Detect pixels that are yellow using the hue component
    hue_binary[(H > 10) & (H <= 25)] = 1
    
    
    return sat_binary,hue_binary


def abs_sobel_thresh(img, sobel_kernel=3, orient='x', thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output


def mag_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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


def combined_threshold(img, kernel=3, grad_thresh=(20,100), mag_thresh=(30,100), dir_thresh=(0, np.pi/2),
                       s_thresh=(90,255), r_thresh=(200,255)):

    def binary_thresh(channel, thresh = (200, 255)):
        binary = np.zeros_like(channel)
        binary[(channel > thresh[0]) & (channel <= thresh[1])] = 1

        return binary

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=kernel, thresh=grad_thresh)
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=kernel, thresh=grad_thresh)
    mag_binary = mag_threshold(img, sobel_kernel=kernel, mag_thresh=mag_thresh)
    dir_binary = dir_threshold(img, sobel_kernel=kernel, thresh=dir_thresh)
    
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]
    sbinary = binary_thresh(S, s_thresh)
    
    R = img[:,:,2]
    G = img[:,:,1]
    B = img[:,:,0]
    rbinary = binary_thresh(R, r_thresh)
    
    combined = np.zeros_like(dir_binary)
    #combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (sbinary ==1 ) | (rbinary == 1)] = 1
    combined[(gradx == 1) | ((mag_binary == 1) & (dir_binary == 1)) | (sbinary ==1 ) | (rbinary == 1)] = 1
    #combined[((gradx == 1) | ((mag_binary == 1)) & (dir_binary == 1)) & ( (sbinary ==1 ) | (rbinary == 1))] = 1
    
    
    return combined

def apply_all_filters(img):
    
    sobel=sobel_x_transform(img)
  
    satur,hue=get_saturation(img)
    
    white=get_white(img)
    
   
    combined = np.zeros_like(sobel)
    combined[(   (sobel == 1) | (satur == 1) | (hue==1)| (white)==1  )] = 1
    
    '''
    f, (ax1, ax2,ax3,ax4,ax5) = plt.subplots(1, 5, figsize=(20,10))
    ax1.imshow(sobel,cmap='gray')
    ax1.set_title('Sobel', fontsize=30)
    ax2.imshow(sobel,cmap='gray')
    ax2.set_title('Saturation', fontsize=30)
    ax3.imshow(satur,cmap='gray')
    ax3.set_title('Hue', fontsize=30)
    ax4.imshow(white,cmap='gray')
    ax4.set_title('white ', fontsize=30)
    ax5.imshow(combined,cmap='gray')
    ax5.set_title('Combined ', fontsize=30)
    
     
    g, (ax5, ax6) = plt.subplots(1, 2, figsize=(20,10))
    
    warped=get_warped_image(combined)
    ax5.imshow(warped,cmap='gray')
    ax5.set_title('Saturation', fontsize=30)

    
    bottom_half = warped[warped.shape[0]//2:,:]

    # TO-DO: Sum across image pixels vertically - make sure to set `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half, axis=0)
    ax6.plot(histogram)
    plt.title('Define peaks on the histogram')
    '''

    return combined

def get_transformed_image(image):
    
    combined=combined_threshold(image)
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(combined,cmap='gray')
    ax2.set_title('Transformed Image', fontsize=30) 








