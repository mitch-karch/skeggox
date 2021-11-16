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

#load dummy image to test
img_color = cv2.imread('img.png')
plt.imshow(img_color)

img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
plt.imshow(img, cmap='gray')

#resize the image
img_color = imutils.resize(img_color, height = 500)
img = imutils.resize(img, height = 500)


#determine the kernel size. we want it to be big enough where the image will be severely blurred so the board will be able to be more defined
k_size = round(np.sqrt(img.shape[0]))*3
#we need an odd kernelsize
if (k_size%2) == 0:
    k_size +=1

#apply median blur to img
#median blur computs the median of al the pixels under the kernel window & the central pixed is replaced with this median value
median_blur = cv2.medianBlur(img,ksize=7)
plt.imshow(median_blur, cmap='gray')
plt.axis('off')


#apply a thresholding function to the median blur first before laplacian
#for every pixel, the same threshold is applied, if the pixel val is smaller than the threshold, it is 0, otherwise the max
#second arg = thresholdd value used to classify pixel values
#third arg = max value used to assign to pixel values exceeding threshold
_, thresh = cv2.threshold(median_blur, 145, 255, cv2.THRESH_BINARY_INV)
plt.imshow(thresh, cmap='gray')
plt.axis('off')

img_edges = cv2.Canny(thresh, 100, 100, apertureSize=3)
plt.imshow(img_edges)
#get the hough lines for the edges
lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

angles = []

for [[x1, y1, x2, y2]] in lines:
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    angles.append(angle)

cv2.imshow("Detected lines", img) 

# median_angle = np.median(angles)
min_angle = min(angles)
img_rotated = ndimage.rotate(img, min_angle)
cv2.imshow('rotated-img', img_rotated)
   
key = cv2.waitKey(0)


print(f"Angle is {min_angle:.04f}")
