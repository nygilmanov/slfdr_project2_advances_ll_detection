## Advanced Lane Lines Detection
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images (Done)
* Apply a distortion correction to raw images (Done)
* Use color transforms, gradients, etc., to create a thresholded binary image.(color filter applied) (Done)
* Apply a perspective transform to rectify binary image ("birds-eye view").(Done)
* Detect lane pixels and fit to find the lane boundary. (Done)
* Determine the curvature of the lane and vehicle position with respect to center.(Curvature is very big!!!)
* Warp the detected lane boundaries back onto the original image.(Done)
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.(lane curvature should be estimated)

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  



### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image( TODO)

The code for this step is contained in the section **"Step1. Pipeline for distortion elimination"** of the IPython notebook located in the root directory  **"./P2_ADV_LL_DETECTION.ipynb"**.
The methods for the camera calibration are located in **Distortion.py** library

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  


![Undistorted](./writeup_images/calibration.png)


I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the **cv2.calibrateCamera()** function.  I applied this distortion correction to the test image using the **cv2.undistort()** function and obtained this result: 

![Undistorted](./writeup_images/undistorted.png)


#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.


Number of functions have been developed to implement the perspective transform ( all the methods are located in **Perspctive.py** library)

- **save_PERSP_to_pickle_file** - saves perspective matrices to the pickle file
- **load_PERSP_from_pickle_file** - loads the perspective matrices from the pickle file
- **generates_perspective_pipeline** - builds trapezoid on straight lines 
- **get_unwarped_image** - gets the original image and returns warped image 

First I defined  trapezoid on the straight lines of the original image (before perspective transform)
For these purposes **DrawLines.py library** has been developed.
It uses techniques for drawing straight lines developed in the first project

![Undistorted](./writeup_images/get_trapezoid.png)


The second step is to define 4 desired points on the image. 
We know that lines are straight on the original image and expect rectangle after transformation

dst =np.float32([[1000,0],[1000,700],[200,0],[200,700]])


Finally we get perspactive matrix and the matrix for the opposite transformation by using **cv2.getPerspectiveTransform**  method

These matrixes are saved in the pickle file and will be used in the later stages of the project.


We can see the result of the transforamation of the original image to the warped one below

![Warped Image](./writeup_images/warped_image.png)

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.


#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate final binary image (thresholding steps at section **Step 3. Gradient and color transformations"** of the main notebook).

The methods used for thresholding are placed in the **Thresholds.py** library


 **Gradients**

- We see that the gradients taken in both the x and the y directions detect the lane lines and pick up other edges. Taking the gradient in the x direction emphasizes edges closer to vertical. Alternatively, taking the gradient in the y direction emphasizes edges closer to horizontal.In our case we need to identify vertical lines. So one of the filters is the gradient by x axis
    
- We also consider the magnitude, or absolute value, of the gradient by x and y axis,which is just the square root of the squares of the individual x and y gradients. For a gradient in both the x and y directions, the magnitude is the square root of the sum of the squares.    
    
- In the case of lane lines, we're interested only in edges of a particular orientation. So we will explore the direction, or orientation, of the gradient as well.


**Color space**

I have explored thresholding individual RGB color channels. 
I have checked them side by side to see which ones do a better job of picking up the lane lines:

    - R = image[:,:,0]
    - G = image[:,:,1]
    - B = image[:,:,2]

Finally i have applyed threshold to the red channel


As a seprate step I have explored thresholding from HLS perspective (hue, lightness, and saturation) and finally
played with saturation
Saturation is a measurement of colorfulness. So, as colors get lighter and closer to white, they have a lower saturation value, whereas colors that are the most intense, like a bright primary color

    - H = hls[:,:,0]
    - L = hls[:,:,1]
    - S = hls[:,:,2]

Finally I have combined all thresholds into one combined expression

combined[(gradx == 1) | ((mag_binary == 1) & (dir_binary == 1)) | (sbinary ==1 ) | (rbinary == 1)] = 1
   

Once we apply the combined transformation to the original image we will get the following result

![Gradient and Color Transformation](./writeup_images/color_gradient.png)


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After applying calibration, thresholding, and a perspective transform to a road image, I  have got a binary image where the lane lines stand out clearly. 
I took a histogram along all the columns in the lower half of the image

With this histogram we are adding up the pixel values along each column in the image. In our thresholded binary image, pixels are either 0 or 1, so the two most prominent peaks in this histogram will be good indicators of the x-position of the base of the lane lines. We can use that as a starting point for where to search for the lines. From that point, we can use a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame.

This process clearly explained here:

https://youtu.be/siAMDK8C_x8

The algorithm looks the following way:

    - Loop through each window in nwindows
    - Find the boundaries of our current window. This is based on a combination of the current window's starting point (leftx_current and rightx_current), as well as the margin you set in the hyperparameters.
    - Use cv2.rectangle to draw these window boundaries onto our visualization image out_img. This is required for the quiz, but you can skip this step in practice if you don't need to visualize where the windows are.
    - Now that we know the boundaries of our window, find out which activated pixels from nonzeroy and nonzerox above actually fall into the window.
    - Append these to our lists left_lane_inds and right_lane_inds.
    - If the number of pixels you found in Step 4 are greater than your hyperparameter minpix, re-center our window (i.e. leftx_current or rightx_current) based on the mean position of these pixels.
    
    
Now that we have found all our pixels belonging to each line through the sliding window method, it's time to fit a polynomial to the line. First, we have a couple small steps to ready our pixels.


In the next frame of video we don't need to do a blind search again, but instead we can just search in a margin around the previous lane line position, like in the above image. The green shaded area shows where we searched for the lines this time. So, once we know where the lines are in one frame of video, we can do a highly targeted search for them in the next frame.


The methods which implement the above mentioned approach are placed  in **"Cureves.py"** library and called in **"Step4. Final pipeline"** section within **final_p** method


Finally I have used the smoothing technique by taking average second order polinomial fit from the 6 mast images.
This technique allowed to improve lane lined detection in the difficult areas.


![alt text](./writeup_images/final_pipeline.png)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

**Radius**

Previously I located the lane line pixels, used their x and y pixel positions to fit a second order polynomial curve described by the following formula

f(y)=Ay^2+By+C


Having this formula we get the formula for the raduis of the curve

![Radius formula](./writeup_images/Formula.png)


The y values of the image increase from top to bottom, so if, for example, we want to measure the radius of curvature closest to the vehicle, we can evaluate the formula above at the y value corresponding to the bottom of your image, or in Python, at yvalue = image.shape[0].


Finally we need to transform the radius values from the pixel space to the real world meters space.

Let's say that our camera image has 720 relevant pixels in the y-dimension (remember, our image is perspective-transformed!), and we'll say roughly 700 relevant pixels in the x-dimension  Therefore, to convert from pixels to real-world meter measurements, we can use:

    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
**Offset**
We can assume the camera is mounted at the center of the car, such that the lane center is the midpoint at the bottom of the image between the two lines you've detected. The offset of the lane center from the center of the image (converted from pixels to meters) is our distance from the center of the lane.    
    
   

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).


Here's a [https://youtu.be/Tjs3X5EyJ1o]

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?


On the challenges videos curves are not idefined accurately.
Here extra playing with color/gradient spaces could make the better job
Alternative  image to image techniques  could also improve the situation (Sanity Check,Reset,Extra Smoothing techniques)






