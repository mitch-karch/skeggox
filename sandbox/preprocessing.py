#imports
#TODO automate the pull from app.box

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import io, img_as_float
import argparse
import imutils

'''
steps to perform:
    - edge detection
    - contour detection
    - four point transformer
        - transform perspective and straighten
'''

#load dummy image to test
img_color = cv2.imread('img.png')
img = cv2.imread('img.png', 0) #load in b&w

#resize the image
img_color = imutils.resize(img_color, height = 500)
img = imutils.resize(img, height = 500)

#equalize the histogram of the grayscale image
eq_img = cv2.equalizeHist(img)
#plot to see the distribution between the original image and the equalized img
plt.hist(img.flat, bins = 100, range=(0,255))
plt.hist(eq_img.flat, bins = 100, range=(0,255))

#show the original image and the equalized img
cv2.imshow('original_color', img_color)
cv2.imshow('original', img)
cv2.imshow('eq_img', eq_img)

#convert the img to lab
lab= cv2.cvtColor(img_color, cv2.COLOR_BGR2LAB)
cv2.imshow("lab",lab)

#convert the img to hsv
hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV )
cv2.imshow("hsv",hsv)

l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)
cv2.imshow('split l', l)
cv2.imshow('CLAHE LAB output', cl)

h, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
clh = clahe.apply(h)
cv2.imshow('split h', h)
cv2.imshow('CLAHE HSV output', clh)

#turn img to float
img_float = img_as_float(img)

#determine the sigma for denoising
sigma_est = np.mean(estimate_sigma(img_float, multichannel=False))

#denoise the image
denoise_img = denoise_nl_means(img_float, h=1.*sigma_est, fast_mode=True,
                              patch_size=5,
                              patch_distance=3,
                              multichannel=False)
cv2.imshow('denoised img', denoise_img)

#standard open cv functions to keep the image window active
cv2.waitKey(0)
cv2.destroyAllWindows()






# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
# kernel = np.ones((5,5),np.float32)/25
# gray = cv2.filter2D(src=gray, ddepth=-1, kernel=kernel)
edged = cv2.Canny(gray, threshold1=20, threshold2=200, apertureSize=3, L2gradient=False)
# show the original image and the edge detected image
print("STEP 1: Edge Detection")
cv2.imshow("Image", img)
cv2.imshow("Edged", edged)

cv2.imshow('gray', gray)

lab= cv2.cvtColor(img_color, cv2.COLOR_BGR2LAB)
cv2.imshow("lab",lab)

l, a, b = cv2.split(lab)
cv2.imshow('l_channel', l)
cv2.imshow('a_channel', a)
cv2.imshow('b_channel', b)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)
cv2.imshow('CLAHE output', cl)

#-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
limg = cv2.merge((cl,a,b))
cv2.imshow('limg', limg)

#-----Converting image from LAB Color model to RGB model--------------------
final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
cv2.imshow('final', final)

cv2.waitKey(0)
cv2.destroyAllWindows()






# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break
# show the contour (outline) of the piece of paper
print("STEP 2: Find contours of paper")
cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", img)
cv2.waitKey(0)
cv2.destroyAllWindows()



alpha = 1.7 # Contrast control (1.0-3.0)
beta = -100 # Brightness control (0-100)

adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
# kernel = np.ones((6,7),np.float32)/25
# gray = cv2.filter2D(src=gray, ddepth=-2, kernel=kernel)
edged = cv2.Canny(gray, threshold1=75, threshold2=200, apertureSize=3, L2gradient=False)
# show the original image and the edge detected image

gray2 = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)
kernel = np.ones((5,5),np.float32)/25
gray2 = cv2.filter2D(src=gray2, ddepth=-2, kernel=kernel)
edged2 = cv2.Canny(gray2, threshold1=75, threshold2=200, apertureSize=3, L2gradient=False)
# show the original image and the edge detected image

cv2.imshow('original', img)
cv2.imshow('adjusted', adjusted)
cv2.imshow('final', final)
cv2.imshow("Edged", edged)
cv2.imshow("Edged2", edged2)
cv2.waitKey()
cv2.destroyAllWindows()

