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
from PIL import Image

from os import listdir, mkdir
from os.path import isfile, join

def normalize_image(img):
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return img.astype(np.uint8)

def resize_image(img):
    return imutils.resize(img, height = 500)

def morphological(thresh):
    #remove excess noise with cv2.MORPH_OPEN
    kernel = np.ones((15,1), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    #close small holes in the foreground with cv2.MORPH_CLOSE
    kernel = np.ones((17,3), np.uint8)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    return morph

def get_edge_points(lines):
    #get the coordinates of all the lines
    coords = list(zip(lines[:,:,0].flatten().tolist(),lines[:,:,1].flatten().tolist()))
    coords.extend(list(zip(lines[:,:,2].flatten().tolist(),lines[:,:,3].flatten().tolist())))
    
    #assume that the edge will never surpass a quarter of the board
    assume_coords = [(x,y) for (x,y) in coords if x < w//4]
    #assume the bottom of the edge will never surpass 480
    assume_coords = [(x,y) for (x,y) in assume_coords if y < 480]
    
    if len(assume_coords) != 0:
        edge_coords = assume_coords
    else:
        edge_coords = coords
    
    #determine initial found edge points
    x1, y1 = min(edge_coords, key=sum)
    x2, y2 = max(edge_coords, key=sum)
    return x1, y1, x2, y2

def get_startpoint(x, y, m, b):
    #point m formula to find the end point if the line segment is too short on top
    while y > h//6:
        x -= 1
        y = int(m*x + b)
        #if y is negative, we want the startpoint to return back to its positive state
        if y < 0:
           x += 1
           y = int(m*x + b)
    return x, y   
    
def get_endpoint(x, y, m, b):
    #point m formula to find the end point if the line segment is too short on bottom     
    while y < h*.9:
        y += 1
        x = int((y/m) - (b/m))
    return x, y

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
    bR_x = bX - vX * length
    bR_y = bY - vY * length
    return int(bR_x), int(bR_y)

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


def write_file(img_to_write):
    cv2.imwrite(join(scale_crop_dir, img_file), img_to_write)
   
def resize_final_img(img):
    if img.shape[0] > 400 or img.shape[1] > 400:
        img = img[0:400,0:400]
    else:
        img = imutils.resize(img, height = 400, width = 400)
    return img

def scale_crop_img(img):
    #apply gaussian blur
    #determine the kernel size. we want it to be big enough where the image will be severely blurred so the board will be able to be more defined
    kernel = (21,21)
    blur = cv2.GaussianBlur(img, kernel, 0)
    
    #apply a thresholding function to the median blur first before laplacian
    #for every pixel, the same threshold is applied, if the pixel val is smaller than the threshold, it is 0, otherwise the max
    thresh = cv2.threshold(blur, 175, 255, cv2.THRESH_BINARY_INV)[1]
    
    #clean up the image's threshold
    morph = morphological(thresh)

    #detect edges
    img_edges = cv2.Canny(morph, 100, 100, apertureSize=7)
    
    #get hough lines from the detected edges
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=30)
    if lines is None: #if no edges can be detected, run it back on the original image
        lines = cv2.HoughLinesP(blur, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=30)
    
    #get the line's edge points to detect the edge of the board
    x1, y1, x2, y2 = get_edge_points(lines)
    
    if x1 > w//3:
        cropped = resize_final_img(img_color)
        write_file(cropped)
        quit
    
    #determine the initial length of the detected edge point
    length = y2-y1
    # angle = math.degrees(math.atan2(y2-y1, x2-x1))
    
    #slope
    if x1-x2 == 0:
        x1 += 1
        
    m = (y1-y2)/(x1-x2)
    
    if m < 1:
        cropped = resize_final_img(img_color)
        write_file(cropped)
        quit
    
    #y-intercept
    b = (x1*y2 - x2*y1)/(x1-x2)
    
    #if the length of the edge detected is less than 90% of the img
    if length < h*.9:
        tL_x, tL_y = get_startpoint(x1, y1, m, b)
        bL_x, bL_y = get_endpoint(x2, y2, m, b)
    
    #determine new length based on expansion of the line
    length = bL_y - tL_y
    
    #get the bottom perpendicular line according to the found edge
    bR_x, bR_y = getPerpCoord(tL_x, tL_y, bL_x, bL_y, length)
    
    #based on the three found coordinates, find the last coordinates of the board
    tR_x = tL_x + (bR_x - bL_x)
    tR_y = tL_y + (bR_y - bL_y)

    #now that we have all four points, perform a four point perspective transform
    pts = [(tL_x, tL_y), (tR_x, tR_y), (bR_x, bR_y), (bL_x, bL_y)]
    pts = np.array(pts, dtype = "float32")
    warped = four_point_transform(img_color, pts)
    cropped = resize_final_img(warped)
    
    cv2.imwrite(join(scale_crop_dir, img_file), cropped)
    
    return cropped

#start program
start_program = time.time()

img_dir = 'grab/imgs/'
img_files = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]

warped_sizes = []

scale_crop_dir = 'imgs_sc'
if not isfile(scale_crop_dir):
    mkdir('imgs_sc')

for img_file in img_files:
    img_path = join(img_dir+img_file)
    # preprocess(img_path)
    
    #load image
    img_color = cv2.imread(img_path)
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    #timing
    start = time.time()
    
    #normalize image
    img = normalize_image(img)
    
    #resize the image
    img_color = resize_image(img_color)
    img = resize_image(img) #bw
    
    h, w = img.shape[:2]
    
    cropped = scale_crop_img(img)
    
    #timing
    end = time.time()
    print(f'Time elapsed: {end-start}')
    
    
    
    # cv2.imshow("Original", img_color) 
    # cv2.imshow("Warped", warped)
    
    # key = cv2.waitKey(0)

end_program = time.time()
print(f'Entire time to process {len(img_files)} images was {end_program-start_program}')



#Testing purposes
def show_img(img):
    cv2.imshow('img', img)
    key = cv2.waitKey(0)
    
def make_line(x1, y1, x2, y2):
    cv2.line(img, (x1,y1), (x2,y2), (255, 0, 0), 5)
