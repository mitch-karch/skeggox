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
    
    
    histr = cv2.calcHist([img],[0],None,[256],[0,256])
    plt.plot(histr)
    
    
    #determine the kernel size. we want it to be big enough where the image will be severely blurred so the board will be able to be more defined
    k_size = round(np.sqrt(img.shape[0]))
    #we need an odd kernelsize
    if (k_size%2) == 0:
        k_size +=1
    
    #apply median blur to img
    #median blur computs the median of al the pixels under the kernel window & the central pixed is replaced with this median value
    # blur = cv2.medianBlur(img,ksize=k_size)
    blur = cv2.GaussianBlur(img, (15,15), 0)
    
    plt.imshow(blur, cmap='gray')
    plt.axis('off')
    
    
    #apply a thresholding function to the median blur first before laplacian
    #for every pixel, the same threshold is applied, if the pixel val is smaller than the threshold, it is 0, otherwise the max
    #second arg = thresholdd value used to classify pixel values
    #third arg = max value used to assign to pixel values exceeding threshold
    _, thresh = cv2.threshold(blur, 145, 255, cv2.THRESH_BINARY_INV)
    
    img_edges = cv2.Canny(blur, 100, 100, apertureSize=3)
    # img_edges = cv2.Canny(thresh, 100, 100, apertureSize=3)
    plt.imshow(img_edges)
    
    #get the hough lines for the edges
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=75, maxLineGap=10)
    
    
    #show the hough lines found
    for [[x1, y1, x2, y2]] in lines:
        # line(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 5)
        # cv2.circle(img, (x1,y1), 3, (0,150,0), -1) #line start
        # cv2.circle(img, (x2,y2), 3, (128,0,0), -1) #line end
        cv2.putText(img, str(x1) + ',' + str(y1), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        cv2.putText(img, str(x2) + ',' + str(y2), (x2,y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        # angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        # angles.append(angle)
    
    
    
    #get the coordinates of all the lines
    coords = list(zip(lines[:,:,0].flatten().tolist(),lines[:,:,1].flatten().tolist()))
    coords.extend(list(zip(lines[:,:,2].flatten().tolist(),lines[:,:,3].flatten().tolist())))
    coords.sort()
    
    # #find the minimum x value and max y value to see where the edge of the board is
    # xmin = int(min(lines[:,:,0].min(axis=0), lines[:,:,2].min(axis=0)))
    # ymax = int(max(lines[:,:,1].max(axis=0), lines[:,:,3].max(axis=0)))
    
    
    # #get the index of the min and max points
    # def find_edge_point(val, val_loc):
    #     if val_loc == 'x':
    #         row, _ , col = np.where(lines[:,:,[0,2]]==val)
    #         if col == 1:
    #             col = 2
    #         point = (int(lines[row,0,col]), int(lines[row,0,col+1]))
    #     else:
    #         row, _ , col = np.where(lines[:,:,[1,3]]==val) 
    #         if col == 1:
    #             col = 3
    #         point = (int(lines[row,0,col-1]), int(lines[row,0,col]))
    #     return point
    
    # x1, y1 = find_edge_point(xmin, val_loc = 'x')
    # x2, y2 = find_edge_point(ymax, val_loc = 'y')
    
    # angle = math.degrees(math.atan2(-y1-y2, x2-x1))
    # h, w = img.shape[:2]
    # center = (w//2, h//2)
    
    # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 3)
    
    #make a line from the min and max points
    cv2.line(img, coords[0], coords[-1], (0, 0, 0), 3)
    
    cv2.imshow("Detected lines", img) 
    
    key = cv2.waitKey(0)


for i in range(0,10):
    img_color = 'imgs/img' + str(i) + '.png'
    draw_detected_edge(img_color)

#determine the other aspects of the board based on our line
length = y2-y1


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
    cX = bX + vX 
    cY = bY + vY 
    dX = bX - vX * length
    dY = bY - vY * length
    return int(cX), int(cY), int(dX), int(dY)

cX, cY, dX, dY = getPerpCoord(x1, y1, x2, y2, length)
cv2.line(img, (cX, cY), (dX, dY), (0, 0, 0), 3)


# cv2.rectangle(img, (x1, y1), (dX, dY), (255,0,0), 6)



# median_angle = np.median(angles)
print(f"Angle is {angle:.04f}")
img_rotated = ndimage.rotate(img, angle)
cv2.imshow('rotated-img', img_rotated)
   

key = cv2.waitKey(0)
