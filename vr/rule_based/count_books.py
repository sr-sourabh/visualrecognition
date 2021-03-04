import sys
import statistics
import cv2 as cv
import numpy as np

sys.setrecursionlimit(15000)
booksScaleDictionary = {"books1.jpg": 0.08, "books2.jpg": 0.1, "books3.jpg": 0.1,
                        "books4.jpg": 0.62, "books5.jpg": 0.6, '12.jpg': 0.1}
bookName = "books1.jpg"
res = 'res/'
img = cv.imread(res + bookName)
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

threshold = cv.adaptiveThreshold(blur, 255, cv.CALIB_CB_ADAPTIVE_THRESH, cv.THRESH_BINARY, 13, 3)
kernel = np.ones((2, 2), np.uint8)
threshold = cv.erode(threshold, kernel, iterations=1)


# Performs flood fill and report height and width of the image
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


# Calculates the number of books
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
