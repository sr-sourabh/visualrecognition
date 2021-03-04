import cv2 as cv

img = cv.imread('/home/sourabh/Pictures/dogimagevr.jpeg')
cv.imshow('Dog', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

average = cv.blur(img, (7, 7))
cv.imshow('Average Blur', average)

gauss = cv.GaussianBlur(img, (7, 7), 3)
cv.imshow('Gauss Blur', gauss)

sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
combine_sobel = cv.bitwise_or(sobelx, sobely)
cv.imshow('Sobel', combine_sobel)

canny = cv.Canny(gray, 130, 130)
cv.imshow('Canny', canny)

cv.waitKey(0)
