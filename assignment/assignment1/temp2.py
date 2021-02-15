import copy
import sys
import statistics
import cv2 as cv
import numpy as np

sys.setrecursionlimit(15000)

dic = {'Rishabh_Pant.jpg': 0.3, 'face2.jpg': 0.4, 'face3.jpg': 0.8,
       'face4.jpg': 0.7, 'face5.jpg': 0.2, 'face6.jpg': 0.6, 'face7.jpg': 0.6, 'face8.jpg': 0.5}
src = 'res/face8.jpg'
img = cv.imread(src)

scale = dic[src]


def rescaleFrame(frame, scale):
    width = int(frame.shape[1] * scale)
    len = int(frame.shape[0] * scale)
    dim = (width, len)
    return cv.resize(frame, dim, interpolation=cv.INTER_AREA)


img = rescaleFrame(img, scale)
coloredImage = img
# cv.imshow('Original', img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
gray = clahe.apply(gray)
# cv.imshow('Gray', gray)

# blur = cv.GaussianBlur(gray, (5, 5), 5)
blur = gray


# canny = cv.Canny(blur, 50, 50)
# cv.imshow('Canny', canny)

def performRangeThreshold(img, lower, upper, val=255):
    width = int(img.shape[1])
    length = int(img.shape[0])

    for i in range(length):
        for j in range(width):
            if lower < img[i][j] < upper:
                img[i][j] = val
            else:
                img[i][j] = 0


kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

# blur = cv.equalizeHist(blur)
rangeThreshold = copy.deepcopy(gray)
performRangeThreshold(rangeThreshold, 140, 235)
# cv.imshow('Blur', blur)
# cv.imshow('InRange', rangeThreshold)

# threshold = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel)
threshold = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 7, 1)
# thresh, threshold = cv.threshold(blur, 100, 255, cv.THRESH_BINARY)
# cv.imshow('ClosedBefore', threshold)
threshold = cv.erode(threshold, kernel, iterations=1)


# cv.imshow('ClosedAfter', threshold)

def bitwiseNand(a, b):
    return cv.bitwise_or(cv.bitwise_and(cv.bitwise_not(a), b), cv.bitwise_and(a, cv.bitwise_not(b)))


# combined = rangeThreshold
combined = bitwiseNand(threshold, rangeThreshold)

# combined = cv.bitwise_or(threshold, rangeThreshold)
cv.imshow('Combined', combined)


def dfs(image, n, m, i, j, height, width, color, area):
    if (i >= n) or (j >= m) or (i < 0) or (j < 0):
        return
    if image[i][j] != 255:
        return
    image[i][j] = color
    dfs(image, n, m, i + 1, j, height, width, color, area)
    dfs(image, n, m, i, j + 1, height, width, color, area)
    dfs(image, n, m, i - 1, j, height, width, color, area)
    dfs(image, n, m, i, j - 1, height, width, color, area)
    height[0] = max(height[0], i)
    height[1] = min(height[1], i)
    width[0] = max(width[0], j)
    width[1] = min(width[1], j)
    area[0] += 1


def floodFill(img, x, y, n, m):
    if (x >= n) or (y >= m) or (x < 0) or (y < 0):
        return
    if img[x][y] == 0:
        return
    if img[x][y] == 240:
        return

    img[x][y] = 240
    floodFill(img, x + 1, y, n, m)
    floodFill(img, x, y + 1, n, m)
    floodFill(img, x - 1, y, n, m)
    floodFill(img, x, y - 1, n, m)


def isValid(h, w, a, imageArea):
    if (0.5 * h < w < 1.1 * h) and (a > (h * w) / 4) and (a < imageArea // 5):
        return True

    return False


def detectFace(img):
    width = int(img.shape[1])
    length = int(img.shape[0])
    imageArea = length * width
    rectangleXDelta = int(20 * scale)
    rectangleYDelta = int(20 * scale)

    print(width, length)

    dimensions = []
    color = 70
    for i in range(length):
        for j in range(width):
            if img[i][j] == 255:
                dim = [[0, 10000], [0, 10000], [0]]
                dfs(img, length, width, i, j, dim[0], dim[1], color, dim[2])
                h = dim[0][0] - dim[0][1]
                w = dim[1][0] - dim[1][1]
                a = dim[2][0]
                if isValid(h, w, a, imageArea):
                    # [height, width, actualArea, begi, begj, minWidthPoint, minHeightPoint, MaxWidthPoint, maxHeightPoint]
                    temp = [h, w, a, i, j, dim[1][1], dim[0][1], dim[1][0], dim[0][0]]
                    dimensions.append(temp)

    dimensions.sort(key=lambda x: x[2], reverse=True)
    print(dimensions)
    dim = dimensions[0]
    pointx = dim[3]
    pointy = dim[4]

    print(dim)
    print(combined[pointx][pointy])
    cv.rectangle(coloredImage, (dim[5] - rectangleXDelta, dim[6] - rectangleYDelta),
                 (dim[7] + rectangleXDelta, dim[8] + rectangleYDelta), (0, 255, 0), thickness=2)
    cv.imshow('FinalWithRectangle', coloredImage)
    floodFill(combined, pointx, pointy, length, width)


detectFace(combined)
cv.imshow('Final', combined)

cv.waitKey(0)
