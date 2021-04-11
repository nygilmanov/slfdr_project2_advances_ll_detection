import math

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            
            print('Coordinates',x1,y1,x2,y2)
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            
            
def draw_lines_extrapolate_modif(img, lines, color=[255, 0, 0], thickness=8):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
        
    ##print('linght function')   
        
        
    ## step1 separate lines to left and right 
    left_lines,right_lines=separate_lines(lines)
    ##print('Left lines',left_lines)
    ##print('Right lines',right_lines)


    ## step2 filter lines by slope
    if right_lines and left_lines:

        #left_lines = reject_outliers(left_lines, cutoff=(-0.85, -0.6))
        #right_lines = reject_outliers(right_lines,  cutoff=(0.45, 0.75))
    
    
        ##print('No right lines before filter:',len(right_lines))
        ##print('No left lines before filter :',len(left_lines))
               
        
        right_lines  = reject_outliers(right_lines,   cutoff=(-1, -0.5))
        left_lines = reject_outliers(left_lines,  cutoff=(0.45, 1))
        
        
        #right_lines  = reject_outliers(right_lines,   cutoff=(-1, -0.1))
        #left_lines = reject_outliers(left_lines,  cutoff=(0.1, 1))
       
        
        
        
        #print('No right lines after filter:', len(right_lines))
        #print('No left lines  after filter :',len(left_lines))

        #print(right_lines,left_lines)
        
        
    y1 = img.shape[0] # bottom of the image
    y2 = y1*0.6         # slightly lower than the middle
    
    
    #print('')
    #print('---------!!!!!!!!     New coordinates------------!!!!!!!',y1,y2)
    
    
    
    x1_r_f,y1_f,x2_r_f,y2_f = get_average_line(right_lines,y1,y2)
    x1_l_f,y1_f,x2_l_f,y2_f = get_average_line(left_lines,y1,y2)
    
    ##x1_r_f,y1_f,x2_r_f,y2_f = lines_linreg(right_lines)
    ##x1_l_f,y1_f,x2_l_f,y2_f = lines_linreg(left_lines)
    
    #print('left  y1=',y1_f,'left x1=',  x1_l_f,'left y2=', y2_f,'left x2=',  x2_l_f)
    #print('right y1=',y1_f,'right x1=', x1_r_f,'right y2=',y2_f,'right x2=', x2_r_f)
   
    
    
    
    ## in case we want linear regression - replace 2 lines above
    
   
    cv2.line(img, (x1_l_f, y1_f), (x2_l_f, y2_f), color, thickness)
    cv2.line(img, (x1_r_f, y1_f), (x2_r_f, y2_f), color, thickness)

    return  [x1_l_f, y1_f,x2_l_f, y2_f],[x1_r_f, y1_f,x2_r_f, y2_f]
    

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    ##draw_lines(line_img, lines)
     
    
        
    #draw_lines_extrapolate(line_img, lines)  
    p1,p2=draw_lines_extrapolate_modif(line_img, lines)  
    
    #plt.imshow(line_img) 
                         
    return line_img,p1,p2


def draw_lines_extrapolate_trapezoid(img, lines, color=[255, 0, 0], thickness=8):

    left_lines,right_lines=separate_lines(lines)
   
    if right_lines and left_lines:
               
        
        right_lines  = reject_outliers(right_lines,   cutoff=(-1, -0.5))
        left_lines = reject_outliers(left_lines,  cutoff=(0.45, 1))
        
        
    y1 = img.shape[0] # bottom of the image
    y2 = y1*0.6         # slightly lower than the middle
    
   
    x1_r_f,y1_f,x2_r_f,y2_f = get_average_line(right_lines,y1,y2)
    x1_l_f,y1_f,x2_l_f,y2_f = get_average_line(left_lines,y1,y2)
    
    
    p1=[x1_l_f, y1_f,x2_l_f, y2_f]
    p2=[x1_r_f, y1_f,x2_r_f, y2_f]
    
    # 0.7 - upper points
    # 0.8 - upper points
    
    # define trapezoid coordinates
    
    y_upper=0.65 
    y_bottom=0.9
    print('------CHECK-------',y_upper,y_bottom)
    
    
    
    
    left_y1=int(img.shape[0]*y_upper)
    left_x1=get_x(left_y1,slope1(p1),p1[2],p1[3])
    left_y2=int(img.shape[0]*y_bottom)
    left_x2=get_x(left_y2,slope1(p1),p1[2],p1[3])
    
    
    
    
    left_x1_hat=left_x2 # this is the real coordinate 
    left_x1_hat2=left_x1 # this is the real coordinate 

    right_y1=int(img.shape[0]*y_upper)
    right_x1=get_x(right_y1,slope1(p2),p2[2],p2[3])
    right_y2=int(img.shape[0]*y_bottom)
    right_x2=get_x(right_y2,slope1(p2),p2[2],p2[3])
    
    right_x1_hat=right_x2 # this is the real coordinate 
    right_x1_hat2=right_x1 
    
    
    left_line_points=[left_x1,left_y1,left_x2,left_y2]
    right_line_points=[right_x1,right_y1,right_x2,right_y2]
    
    
    
    
    print('Point1=',p1)
    print('Point2=',p2)

    print('LeftPoint1:','y1=',left_y1,'x1=',left_x1)
    print('LeftPoint2:','y2=',left_y2,'x2=',left_x2)
    
    
    print('RightPoint1:','y1=',right_y1,'x1=',right_x1)
    print('RightPoint2:','y2=',right_y2,'x2=',right_x2)
    
    
    
    print('LeftPoint:','y1=',left_y1,'x_hat=',left_x1_hat)
    print('RightPoint:','y1=',right_y1,'x_hat=',right_x1_hat)
    
    #print(right_y1,right_x1,right_y2,right_x2)
    
    
        
    cv2.line(img, (left_x1, left_y1),   (left_x2, left_y2)  , color, thickness)
    cv2.line(img, (right_x1, right_y1), (right_x2, right_y2), color, thickness)
    cv2.line(img, (left_x1, left_y1),   (right_x1, right_y1), color, thickness)
    cv2.line(img, (left_x2, left_y2),   (right_x2, right_y2), color, thickness)
    
    
    color2=[0, 0, 255] 
    cv2.line(img, (left_x1_hat, left_y1),   (left_x2, left_y2)  , color2, thickness)
    cv2.line(img, (right_x1_hat, right_y1), (right_x2, right_y2)  , color2, thickness)
   
    #-------------
    color3=[0, 255, 0] 
    cv2.line(img, (left_x1, left_y1),   (left_x1_hat2, left_y2)  , color3, thickness)
    cv2.line(img, (right_x1, right_y1), (right_x1_hat2, right_y2)  , color3, thickness)
    
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, str(left_x1)+'--'+str(left_y1), (left_x1, left_y1), font, 1, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(img, str(left_x2)+'--'+str(left_y2), (left_x2, left_y2), font, 1, (255,255,255), 2, cv2.LINE_AA)
    
    cv2.putText(img, str(right_x1)+'--'+str(right_y1), (right_x1, right_y1), font, 1, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(img, str(right_x2)+'--'+str(right_y2), (right_x2, left_y2), font, 1, (255,255,255), 2, cv2.LINE_AA)
    
    
    
    cv2.putText(img, str(left_x1_hat)+'--'+str(left_y1), (left_x1_hat, left_y1), font, 1, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(img, str(right_x1_hat)+'--'+str(right_y1), (right_x1_hat, right_y1), font, 1, (255,255,255), 2, cv2.LINE_AA)
    
    cv2.putText(img, str(left_x1_hat2)+'--'+str(left_y2), (left_x1_hat2, left_y2), font, 1, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(img, str(right_x1_hat2)+'--'+str(right_y2), (right_x1_hat2, right_y2), font, 1, (255,255,255), 2, cv2.LINE_AA)
    
    
    
    
    #cv2.putText(img, str(left_x1)+''+str(left_y1), (left_x1, left_y1), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    
   
   
    p1_hat=[left_x1_hat, left_y1,left_x2, left_y2]
    p2_hat=[right_x1_hat, right_y1,right_x2, right_y2]
    
    #p1,p2  actual on the image
    #p1_hat,p2_hat realy parallel lines 
    
    return left_line_points,right_line_points,p1_hat,p2_hat
    
    
def hough_lines_trapezoid(img, rho, theta, threshold, min_line_len, max_line_gap):

    """
    function  
    1.gets hough lines
    2.calculates average left and right lines
    3.based on the lines moves coordinates to get right trapezoid
    4.drawst rapezoid
    
    """
    
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    ##draw_lines(line_img, lines)
     
    
        
    # return real and expected trapezoid coodrinates 
    p1,p2,p1_hat,p2_hat=draw_lines_extrapolate_trapezoid(line_img,lines)  
    
    #plt.imshow(line_img) 
                         
    return line_img,p1,p2,p1_hat,p2_hat


def draw_trapezoid(img, p1,p2, color=[255, 0, 0], thickness=8):
    
 
    left_y1=int(img.shape[0]*0.7)
    left_x1=get_x(left_y1,dll.slope1(p1),p1[2],p1[3])
    left_y2=int(img.shape[0]*0.9)
    left_x2=get_x(left_y2,dll.slope1(p1),p1[2],p1[3])

    right_y1=int(img.shape[0]*0.7)
    right_x1=get_x(right_y1,dll.slope1(p2),p2[2],p2[3])
    right_y2=int(img.shape[0]*0.9)
    right_x2=get_x(right_y2,dll.slope1(p2),p2[2],p2[3])

    print(left_y1,left_x1,left_y2,left_x2)
    print(right_y1,right_x1,right_y2,right_x2)
    
    
 
    cv2.line(img, (left_x1, left_y1),   (left_x2, left_y2)  , color, thickness)
    cv2.line(img, (right_x1, right_y1), (right_x2, right_y2), color, thickness)
    cv2.line(img, (left_x1, left_y1),   (right_x1, right_y1), color, thickness)
    cv2.line(img, (left_x2, left_y2),   (right_x2, right_y2), color, thickness)



def hough_curves(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    ##lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    
    lines = cv2.HoughLinesP(img,cv2.HOUGH_PROBABILISTIC, np.pi/180, 30, minLineLength=min_line_len,maxLineGap=max_line_gap)

    
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img



# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    #plt.imshow(initial_img) 
    #plt.show()
    ##plt.imshow(cv2.addWeighted(initial_img, α, img, β, γ)) 
    ##plt.show()
    
    
    return cv2.addWeighted(initial_img, α, img, β, γ)



def slope(x1, y1, x2, y2):
    return (y1 - y2) / (x1 - x2)

def slope1(lst):
    return (lst[1] - lst[3]) / (lst[0] - lst[2])

def get_x(y1,slope,x2,y2):
    x1=(y1-y2)/slope+x2
    return int(x1)


def separate_lines(lines):
    right = []
    left = []
    for x1,y1,x2,y2 in lines[:, 0]:
        m = slope(x1,y1,x2,y2)
        if m >= 0:
            right.append([x1,y1,x2,y2,m])
        else:
            left.append([x1,y1,x2,y2,m])
    return right, left


def reject_outliers(data, cutoff, threshold=0.1):
    data = np.array(data)
    #print('Reject outliers input data',data)
    #print('DATA[:,4]',data[:, 4])
    
    
    #print('CUTOFF VALUES', cutoff[0],cutoff[1])
    
    data = data[(data[:, 4] >= cutoff[0]) & (data[:, 4] <= cutoff[1])]
    m = np.mean(data[:, 4], axis=0)
    
    #print('m=',m,'m+threshold',m+threshold,'m-threshold',m-threshold)
    
    return data
          
    ##return data[(data[:, 4] <= m+threshold) & (data[:, 4] >= m-threshold)]


def get_average_line(lines,y1_ext,y2_ext):

    dt=np.array(lines)
    
    #print('lines input',dt)
    if len(dt)==0:
        print('Input is empty')
    
    
    x = np.reshape(dt[:, [0, 2]], (1, len(dt) * 2))[0]
    y = np.reshape(dt[:, [1, 3]], (1, len(dt) * 2))[0]
    slope_hat=np.mean((dt[:,3]-dt[:,1])/(dt[:,2]-dt[:,0]))

    x_hat=np.mean(x)
    y_hat=np.mean(y)

    y1=int(y1_ext)  ## take one coordinate on the top
    y2=int(y2_ext)  ## take one coordinate on the bottom

    
    #print('x_hat=',x_hat,'y_hat=',y_hat)
    

    x1= int((y1-y_hat)/slope_hat+x_hat)         
    x2= int((y2-y_hat)/slope_hat+x_hat)

    return x1,y1,x2,y2


def lines_linreg(lines_array):
    x = np.reshape(lines_array[:, [0, 2]], (1, len(lines_array) * 2))[0]
    y = np.reshape(lines_array[:, [1, 3]], (1, len(lines_array) * 2))[0]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    x = np.array(x)
    y = np.array(x * m + c)
    
    
    min_y = np.min(y)
    # Calculate the top point using the slopes and intercepts we got from linear regression.
    top_point = np.array([(min_y - c) / m, min_y], dtype=int)
    
    #print('top_point',top_point)
    
    # Repeat this process to find the bottom left point.
    max_y = np.max(y)
    bot_point = np.array([(max_y - c) / m, max_y], dtype=int)
    
    #print('bot_point',bot_point)
    

    y1 = min_y
    x1 = (min_y - c) / m
    
    y2 = max_y
    x2 = (max_y - c) / m
    
    return int(x1),int(y1),int(x2),int(y2)

def filter_image(image):
    hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    sensitivity = 95
    lower_white = np.array([0,0,255-sensitivity])
    upper_white = np.array([255,sensitivity,255])
    lower_yellow = np.array([18,102,204], np.uint8)
    upper_yellow = np.array([25,255,255], np.uint8)
    white_mask =cv2.inRange(hsv_image, lower_white, upper_white)
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    filtered_image = cv2.bitwise_and(image, image, mask=white_mask+yellow_mask)
    return filtered_image

def convert_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

def convert_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

def select_white_yellow(image):
    converted = convert_hls(image)
    # white color mask
    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    # yellow color mask
    lower = np.uint8([ 10,   0, 100])
    upper = np.uint8([ 40, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask = mask)

def get_vertices(image):
    """
    It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
    """
    # first, define the polygon by vertices
    rows, cols = image.shape[:2]
    bottom_left  = [cols*0.1, rows*0.95]
    top_left     = [cols*0.4, rows*0.6]
    bottom_right = [cols*0.9, rows*0.95]
    top_right    = [cols*0.6, rows*0.6] 
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return vertices

def select_white_yellow(image):
    converted = convert_hls(image)
    # white color mask
    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    # yellow color mask
    lower = np.uint8([ 10,   0, 100])
    upper = np.uint8([ 40, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask = mask)







