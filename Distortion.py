'''
library contains methods to fit poli lines 

'''

import imp as mp
import DrawLaneLines as dll
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import pickle
import pandas as pd
from math import ceil


def plot_images_from_folder(images):
    
    
    plt.subplots(figsize=(10,30)) # optional
    cols = 2
    rows = ceil(len(images) / cols)

    print(cols,rows)

    # iterate through indices and keys
    for index, key in enumerate(images):
        #img = cv2.imread(key)
        img = mpimg.imread(key)
        ##print(img.shape)
        plt.subplot(rows, cols,index + 1) 
        plt.imshow(img)    
        plt.title(key)

    # render everything
    plt.subplots_adjust(wspace=None, hspace=None)
    
    
def save_to_mtx_dist_pickle_file(mtx,dist):
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open("./camera_cal/wide_dist_pickle.p", "wb" ))    
    
    
def get_mtx_dist_from_pickle_file():
    ob = pd.read_pickle("./camera_cal/wide_dist_pickle.p")
    mtx = ob["mtx"]
    dist = ob["dist"]
    return mtx,dist   

def cal_undistort(img,mtx,dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def get_distortion_mts_dist():

    '''
    1.generate obect points - all the same for each case
    2.generate image points - different for different images
    3.
    '''

    images=glob.glob('./camera_cal/calibration*.jpg')

    objpoints=[]
    imgpoints=[]
    objp = np.zeros((6*9,3),np.float32)

    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) # desired images shape 

    for fname in images:
        ##print(fname)
        img = cv2.imread(fname)
        gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        print(fname,ret)
        if ret == True:
            # for every new image 
            imgpoints.append(corners)
            objpoints.append(objp)

            # Draw and display the corners
            img=cv2.drawChessboardCorners(img, (9,6), corners, ret)
            plt.imshow(img) 
            plt.show()

            
    ##get any image do define the image size
    
    img = cv2.imread('./camera_cal/calibration3.jpg')
    img_size = (img.shape[1], img.shape[0])
    print(img_size)
    print(img.shape[1:])
    
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    
    ## save results to pickle file
    save_to_mtx_dist_pickle_file(mtx,dist)
    
def get_distortion_mts_dist_mod():

    '''
    1.generate obect points - all the same for each case
    2.generate image points - different for different images
    3.
    '''

    images=glob.glob('./camera_cal/calibration*.jpg')

    objpoints=[]
    imgpoints=[]
    objp = np.zeros((6*9,3),np.float32)

    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) # desired images shape 
    
    
    
    plt.subplots(figsize=(15,40)) # optional
    cols = 2
    rows = ceil(len(images) / cols)
    print(cols,rows)
    
    
    index=0 

    for fname in images:
        ##print(fname)
        img = cv2.imread(fname)
        gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        #print(fname,ret)
        if ret == True:
            
            
            # for every new image 
            imgpoints.append(corners)
            objpoints.append(objp)

            # Draw and display the corners
            img=cv2.drawChessboardCorners(img, (9,6), corners, ret)
            #plt.imshow(img) 
            #plt.show()
            
            plt.subplot(rows, cols,index + 1) 
            plt.imshow(img)  
            plt.title(fname)
      
            index+=1

            
    ##get any image do define the image size
    
    img = cv2.imread('./camera_cal/calibration3.jpg')
    img_size = (img.shape[1], img.shape[0])
    #print(img_size)
    #print(img.shape[1:])
    
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    
    ## save results to pickle file
    save_to_mtx_dist_pickle_file(mtx,dist)    
    
    
def print_orig_undistorted(fname):
    
    ##img=cv2.imread(fname)
    
    img = mpimg.imread(fname)
    
    mtx,dist=get_mtx_dist_from_pickle_file()
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(undist)
    ax2.set_title('Undistorted Image', fontsize=30)
    


    
    
    
    
    
    
    
    
    
    
    
    