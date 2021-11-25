# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 17:45:19 2021

@author: marcianolynn
"""

#imports
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import io, img_as_float
import argparse
import imutils
import math
from scipy import ndimage
from scipy import stats



def draw_detected_edge(img_color):
    #load dummy image to test
    img_color = cv2.imread(img_color)
    plt.imshow(img_color)
    
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    plt.imshow(img, cmap='gray')
    
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
    
    #apply median blur to img
    #median blur computs the median of al the pixels under the kernel window & the central pixed is replaced with this median value
    # blur = cv2.medianBlur(img,ksize=k_size)
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
    
    
    #show the hough lines found
    for [[x1, y1, x2, y2]] in lines:
    #     # line(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 5)
    #     # cv2.circle(img, (x1,y1), 3, (0,150,0), -1) #line start
    #     # cv2.circle(img, (x2,y2), 3, (128,0,0), -1) #line end
        cv2.putText(img, str(x1) + ',' + str(y1), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        cv2.putText(img, str(x2) + ',' + str(y2), (x2,y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
    #     # angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    #     # angles.append(angle)
    
    
    
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
    
    
    angle = math.degrees(math.atan2(y2-y1, x2-x1))
    
    #slope
    slope = (y1-y2)/(x1-x2)
    #y-intercept
    b = (x1*y2 - x2*y1)/(x1-x2)
    
    print(f'm={slope}, b={b}')
    
    #y = m*x + b
    def get_startpoint(x, y):
        #point slope formula to find the end point if the line segment is too short on top
        while y > h//6:
            print('minpoint', x, y)
            x -= int(slope/2)
            y = int(slope*x + b)
            if y < 0:
               x += int(slope*2) 
               y = int(slope*x + b)
        return x, y   
        
    def get_endpoint(x, y):
        #point slope formula to find the end point if the line segment is too short on bottom     
        while y < h//1.1:
            print('maxpoint', x,y)
            y += int(slope)
            x = int((y/slope) - (b/slope))
        return x, y
    
    
    
    if length < h//1.1:
        x_start, y_start = get_startpoint(x1, y1)
        
        x_end, y_end = get_endpoint(x2, y2)

            
    
    # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 3)
    
    #make a line from the min and max points
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 3)
    
    
    # cv2.putText(img, 'startpoint', (x_start,y_start), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
    # cv2.putText(img, 'endpoint', (x_end,y_end), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
    
    #make a line from the min and max points
    cv2.line(img, (x_start,y_start), (x_end,y_end), (255, 0, 0), 5)
    
    cv2.imshow("Detected lines", img) 
    
    key = cv2.waitKey(0)


for i in range(0,10):
    img_color = 'imgs/img' + str(i) + '.png'
    draw_detected_edge(img_color)







# #determine the other aspects of the board based on our line
# length = y2-y1


# def getPerpCoord(aX, aY, bX, bY, length):
#     vX = bX-aX
#     vY = bY-aY
#     #print(str(vX)+" "+str(vY))
#     if(vX == 0 or vY == 0):
#         return 0, 0, 0, 0
#     mag = math.sqrt(vX*vX + vY*vY)
#     vX = vX / mag
#     vY = vY / mag
#     temp = vX
#     vX = 0-vY
#     vY = temp
#     cX = bX + vX 
#     cY = bY + vY 
#     dX = bX - vX * length
#     dY = bY - vY * length
#     return int(cX), int(cY), int(dX), int(dY)

# cX, cY, dX, dY = getPerpCoord(x1, y1, x2, y2, length)
# cv2.line(img, (cX, cY), (dX, dY), (0, 0, 0), 3)


# # cv2.rectangle(img, (x1, y1), (dX, dY), (255,0,0), 6)



# # median_angle = np.median(angles)
# print(f"Angle is {angle:.04f}")
# img_rotated = ndimage.rotate(img, angle)
# cv2.imshow('rotated-img', img_rotated)
   

# key = cv2.waitKey(0)

