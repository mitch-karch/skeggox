import numpy as np
import cv2

img = cv2.imread('input copy.png')
windowName = 'image'

minDist = 300
param1 = 30
param2 = 10 #200 #smaller value-> more false circles
minRadius = 180
maxRadius = 210#10

def hough():
    global minDist,param1,param2,minRadius,maxRadius
    
    print(f"{minDist},{param1},{param2},{minRadius},{maxRadius}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 15) #cv2.bilateralFilter(gray,10,50,50)

    # docstring of HoughCircles: HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    imageCopy = blurred.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(imageCopy, (i[0], i[1]), i[2], (0, 255, 0), 5)

    cv2.imshow(windowName, imageCopy)


def on_change(val):
    global minDist
    minDist = val
    hough()

def on_change2(val):
    global param1
    param1 = val
    hough()

def on_change3(val):
    global param2
    param2 = val
    hough()

def on_change4(val):
    global minRadius
    minRadius= val
    hough()

def on_change5(val):
    global maxRadius
    maxRadius = val
    hough()


cv2.namedWindow('controls')
# Show result for testing:
cv2.imshow(windowName, img)
cv2.createTrackbar('minDist', 'controls', minDist, 500, on_change)
cv2.createTrackbar('param1','controls' , param1, 500, on_change2)
cv2.createTrackbar('param2','controls' , param2, 500, on_change3)
cv2.createTrackbar('minRadius', 'controls', minRadius, 500, on_change4)
cv2.createTrackbar('maxRadius', 'controls', maxRadius, 500, on_change5)

cv2.waitKey(0)
cv2.destroyAllWindows()