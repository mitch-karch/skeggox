"""
Created on Mon Nov 15 17:45:19 2021

@author: marcianolynn
"""

#imports
import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
import math
import time


def draw_detected_edge(img_color):    
    #load dummy image to test
    img_color = cv2.imread(img_color)
    plt.imshow(img_color)
    
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    plt.imshow(img, cmap='gray')
    
    #timing
    start = time.time()
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = img.astype(np.uint8)
    plt.imshow(img, cmap='gray')
    
    #resize the image
    img_color = imutils.resize(img_color, height = 500)
    img = imutils.resize(img, height = 500)
    
    h, w = img.shape[:2]
    
    histr = cv2.calcHist([img],[0],None,[256],[0,256])
    plt.plot(histr)
    
    
    #determine the kernel size. we want it to be big enough where the image will be severely blurred so the board will be able to be more defined
    k_size = round(np.sqrt(img.shape[0]))
    #we need an odd kernelsize
    if (k_size%2) == 0:
        k_size +=1
    
    kernel = (21,21)
    
    #apply gaussian blur
    blur = cv2.GaussianBlur(img, kernel, 0)
    
    plt.imshow(blur, cmap='gray')
    plt.axis('off')
    

    
    #apply a thresholding function to the median blur first before laplacian
    #for every pixel, the same threshold is applied, if the pixel val is smaller than the threshold, it is 0, otherwise the max
    #second arg = thresholdd value used to classify pixel values
    #third arg = max value used to assign to pixel values exceeding threshold
    _, thresh = cv2.threshold(blur, 175, 255, cv2.THRESH_BINARY_INV)
    
    # apply close to connect the white areas
    kernel = np.ones((15,1), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((17,3), np.uint8)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow("morph", morph)
    
    
    img_edges = cv2.Canny(morph, 100, 100, apertureSize=7)
    # img_edges = cv2.Canny(thresh, 100, 100, apertureSize=3)
    plt.imshow(img_edges)
    
    #get the hough lines for the edges
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=20)
    
    
    #get the coordinates of all the lines
    coords = list(zip(lines[:,:,0].flatten().tolist(),lines[:,:,1].flatten().tolist()))
    coords.extend(list(zip(lines[:,:,2].flatten().tolist(),lines[:,:,3].flatten().tolist())))
    
    #assume that the edge will never surpass a quarter of the board
    coords = [(x,y) for (x,y) in coords if x < w//4]
    #assume the bottom of the edge will never surpass 480
    coords = [(x,y) for (x,y) in coords if y < 490]
    
    #determine initial found edge points
    x1, y1 = min(coords, key=sum)
    x2, y2 = max(coords, key=sum)
    
    length = y2-y1
    
    
    # angle = math.degrees(math.atan2(y2-y1, x2-x1))
    
    #slope
    slope = (y1-y2)/(x1-x2)
    #y-intercept
    b = (x1*y2 - x2*y1)/(x1-x2)
        
    def get_startpoint(x, y):
        #point slope formula to find the end point if the line segment is too short on top
        while y > h//6:
            x -= 1
            y = int(slope*x + b)
            #if y is negative, we want the startpoint to return back to its positive state
            if y < 0:
               x += 1
               y = int(slope*x + b)
        return x, y   
        
    def get_endpoint(x, y):
        #point slope formula to find the end point if the line segment is too short on bottom     
        while y < h*.9:
            y += 1
            x = int((y/slope) - (b/slope))
        return x, y
    
    #if the length of the edge detected is less than 90% of the img
    if length < h*.9:
        x_start, y_start = get_startpoint(x1, y1)
        x_end, y_end = get_endpoint(x2, y2)

    
    #make a line from the start and end points
    # cv2.line(img, (x_start,y_start), (x_end,y_end), (255, 0, 0), 5)
    
    new_length = y_end - y_start
    
    #get the perpendicular line
    def getPerpCoord(aX, aY, bX, bY, length):
        vX = bX-aX
        vY = bY-aY
        #print(str(vX)+" "+str(vY))
        if(vX == 0 or vY == 0):
            return 0, 0, 0, 0
        mag = math.sqrt(vX*vX + vY*vY)
        vX = vX / mag
        vY = vY / mag
        temp = vX
        vX = 0-vY
        vY = temp
        dX = bX - vX * length
        dY = bY - vY * length
        return int(cX), int(cY), int(dX), int(dY)
    
    cX, cY, dX, dY = getPerpCoord(x_start,y_start, x_end,y_end, new_length+20)
    
    tX = x_start + (dX - x_end)
    tY = y_start + (dY - y_end)

    
    def order_points(pts):
    	# initialzie a list of coordinates that will be ordered
    	# such that the first entry in the list is the top-left,
    	# the second entry is the top-right, the third is the
    	# bottom-right, and the fourth is the bottom-left
    	rect = np.zeros((4, 2), dtype = "float32")
    	# the top-left point will have the smallest sum, whereas
    	# the bottom-right point will have the largest sum
    	s = pts.sum(axis = 1)
    	rect[0] = pts[np.argmin(s)]
    	rect[2] = pts[np.argmax(s)]
    	# now, compute the difference between the points, the
    	# top-right point will have the smallest difference,
    	# whereas the bottom-left will have the largest difference
    	diff = np.diff(pts, axis = 1)
    	rect[1] = pts[np.argmin(diff)]
    	rect[3] = pts[np.argmax(diff)]
    	# return the ordered coordinates
    	return rect
    
    def four_point_transform(image, pts):
    	# obtain a consistent order of the points and unpack them
    	# individually
    	rect = order_points(pts)
    	(tl, tr, br, bl) = rect
    	# compute the width of the new image, which will be the
    	# maximum distance between bottom-right and bottom-left
    	# x-coordiates or the top-right and top-left x-coordinates
    	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    	maxWidth = max(int(widthA), int(widthB))
    	# compute the height of the new image, which will be the
    	# maximum distance between the top-right and bottom-right
    	# y-coordinates or the top-left and bottom-left y-coordinates
    	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    	maxHeight = max(int(heightA), int(heightB))
    	# now that we have the dimensions of the new image, construct
    	# the set of destination points to obtain a "birds eye view",
    	# (i.e. top-down view) of the image, again specifying points
    	# in the top-left, top-right, bottom-right, and bottom-left
    	# order
    	dst = np.array([
    		[0, 0],
    		[maxWidth - 1, 0],
    		[maxWidth - 1, maxHeight - 1],
    		[0, maxHeight - 1]], dtype = "float32")
    	# compute the perspective transform matrix and then apply it
    	M = cv2.getPerspectiveTransform(rect, dst)
    	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    	# return the warped image
    	return warped
    
    pts = [(x_start, y_start), (tX, tY), (dX, dY), (x_end, y_end)]
    pts = np.array(pts, dtype = "float32")
    warped = four_point_transform(img, pts)
    
    #timing
    end = time.time()
    print(f'Time elapsed: {end-start}')
    
    cv2.imshow("Detected lines", img) 
    cv2.imshow("warped", warped)
    
    key = cv2.waitKey(0)


for i in range(0,10):
    img_color = 'imgs/img' + str(i) + '.png'
    draw_detected_edge(img_color)


