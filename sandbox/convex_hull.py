import numpy as np
import cv2

img = cv2.imread('input copy.png')
windowName = 'image'

def on_change(val):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 15) #cv2.bilateralFilter(gray,10,50,50)

    minDist = 300
    param1 = val
    param2 = 10 #200 #smaller value-> more false circles
    minRadius = 180
    maxRadius = 210#10

    # docstring of HoughCircles: HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    imageCopy = blurred.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(imageCopy, (i[0], i[1]), i[2], (0, 255, 0), 5)


    cv2.putText(imageCopy, str(val), (0, imageCopy.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 4)
    cv2.imshow(windowName, imageCopy)

# Show result for testing:
cv2.imshow(windowName, img)
cv2.createTrackbar('slider', windowName, 80, 500, on_change)

cv2.waitKey(0)
cv2.destroyAllWindows()