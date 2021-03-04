import cv2 as cv
import numpy as np

cv.ocl.setUseOpenCL(False)

imageScaleDictionary = {'panaroma1.jpg': 0.5, 'panaroma2.jpg': 0.5}
res = 'res/'
imageName1 = 'panaroma1.jpg'
img1 = cv.imread(res + imageName1)
scale1 = imageScaleDictionary[imageName1]
imageName2 = 'panaroma2.jpg'
img2 = cv.imread(res + imageName2)
scale2 = imageScaleDictionary[imageName2]


def rescaleFrame(frame, scale):
    width = int(frame.shape[1] * scale)
    len = int(frame.shape[0] * scale)
    dim = (width, len)
    return cv.resize(frame, dim, interpolation=cv.INTER_AREA)


img1 = rescaleFrame(img1, scale1)
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = rescaleFrame(img2, scale2)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# cv.imshow('gray1', gray1)
# cv.imshow('gray2', gray2)

keypoints1, feature1 = cv.ORB_create().detectAndCompute(gray1, None)
keypoints2, feature2 = cv.ORB_create().detectAndCompute(gray2, None)

# cv.imshow('Panaroma1', cv.drawKeypoints(gray1, keypoints1, None, color=(0, 255, 0)))
# cv.imshow('Panaroma2', cv.drawKeypoints(gray2, keypoints2, None, color=(0, 255, 0)))

match = cv.BFMatcher()
matches = match.knnMatch(feature1, feature2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.6 * n.distance:
        good.append(m)

draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, flags=2)
img3 = cv.drawMatches(img1, keypoints1, img2, keypoints2, good, None, **draw_params)
cv.imshow("Good Matches", img3)

keypoints1 = np.float32([kp.pt for kp in keypoints1])
keypoints2 = np.float32([kp.pt for kp in keypoints2])
pts1 = np.float32([keypoints1[m.queryIdx] for m in good])
pts2 = np.float32([keypoints2[m.trainIdx] for m in good])

H, status = cv.findHomography(pts1, pts2, cv.RANSAC, 4)

width = img1.shape[1] + img2.shape[1]
height = img1.shape[0] + img2.shape[0]

result = cv.warpPerspective(img1, H, (width, height))
result[0:img2.shape[0], 0:img2.shape[1]] = img2
cv.imshow('Final', result)

cv.waitKey(0)
