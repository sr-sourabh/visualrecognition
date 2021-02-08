import sys
import statistics
import cv2 as cv
import numpy as np

sys.setrecursionlimit(15000)
booksScaleDictionary = {"books1.jpg": 0.08, "books2.jpg": 0.1, "books3.jpg": 0.1,
                        "books4.jpg": 0.62, "books5.jpg": 0.6, '12.jpg': 0.1}
bookName = "12.jpg"
img = cv.imread(bookName)
scale = booksScaleDictionary[bookName]


def rescaleFrame(frame, scale):
    width = int(frame.shape[1] * scale)
    len = int(frame.shape[0] * scale)
    dim = (width, len)
    return cv.resize(frame, dim, interpolation=cv.INTER_AREA)


img = rescaleFrame(img, scale)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)
blur = cv.GaussianBlur(gray, (3, 3), 9)
# cv.imshow('BookBlur', blur)

# canny = cv.Canny(img, 90, 90)
# cv.imshow('Canny', canny)

# sobelx = cv.Sobel(blur, cv.CV_64F, 1, 0)
# sobely = cv.Sobel(blur, cv.CV_64F, 0, 1)
# sobel = cv.bitwise_or(sobelx, sobely)

# cv.imshow('Sobelx', sobelx)
# cv.imshow('Sobely', sobely)
# cv.imshow('Sobel', sobel)

threshold = cv.adaptiveThreshold(blur, 255, cv.CALIB_CB_ADAPTIVE_THRESH, cv.THRESH_BINARY, 13, 3)

# cv.imshow('Adaptive Thresholding', threshold)

# kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
# closed = cv.morphologyEx(threshold, cv.MORPH_CLOSE, kernel)
kernel = np.ones((2, 2), np.uint8)
threshold = cv.erode(threshold, kernel, iterations=1)


# cv.imshow('threshold', threshold)
# cv.imshow('Closed', closed)

# threshold = cv.Canny(gray, 80, 90)
# threshold, thresh = cv.threshold(blur, 130, 255, cv.THRESH_BINARY)
# cv.imshow('Normal Thresholding', thresh)

def dfs(image, n, m, begi, begj, i, j, height, width, color):
    if (i >= n) or (j >= m) or (i < 0) or (j < 0):
        return
    if image[i][j] != 255:
        return
    image[i][j] = color
    dfs(image, n, m, begi, begj, i + 1, j, height, width, color)
    dfs(image, n, m, begi, begj, i, j + 1, height, width, color)
    dfs(image, n, m, begi, begj, i - 1, j, height, width, color)
    dfs(image, n, m, begi, begj, i, j - 1, height, width, color)
    height[0] = max(height[0], abs(begi - i))
    width[0] = max(width[0], abs(begj - j))


def calBookNumber(img):
    width = int(img.shape[1])
    length = int(img.shape[0])

    print(width, length)

    dimensions = []
    areas = []
    heightThreshold = 2
    widthThreshold = 2
    bookAspectRatio = 3
    color = 200
    for i in range(length):
        for j in range(width):
            if img[i][j] == 255:
                dim = [[0], [0]]
                dfs(img, length, width, i, j, i, j, dim[0], dim[1], color)
                h = dim[0][0]
                w = dim[1][0]
                if (h > bookAspectRatio * w or w > h * bookAspectRatio) \
                        and h > heightThreshold and w > widthThreshold:
                    areas.append(dim[0][0] * dim[1][0])
                    dimensions.append(dim)

    bookAreasMedian = 0
    minBookThreshold = 10
    if areas:
        bookAreasMedian = statistics.median(areas)
        print('Median: ', bookAreasMedian)
    print(len(dimensions))
    if len(dimensions) >= minBookThreshold:
        dimensions = [x for x in dimensions if x[0][0] * x[1][0] > bookAreasMedian]
    print(len(dimensions))


cv.imshow('before', threshold)
calBookNumber(threshold)
cv.imshow('after', threshold)

cv.waitKey(0)
