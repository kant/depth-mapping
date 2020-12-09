import cv2
import numpy as np
import matplotlib.pyplot as plt

I1 = cv2.imread('./data/myL.jpeg');
I2 = cv2.imread('./data/myR.jpeg');

I1gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
I2gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(I1gray, None)
kp2, des2 = orb.detectAndCompute(I2gray, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)[:30]

img3 = cv2.drawMatches(I1,kp1,I2,kp2, matches, flags=2, outImg = None)
plt.imshow(img3), plt.show()

best_kp1 = []
best_kp2 = []
best_matches = []

for match in matches:
	best_kp1.append(kp1[match.queryIdx].pt)
	best_kp2.append(kp2[match.trainIdx].pt)
	best_matches.append(match)

print(best_kp1)
print(best_kp2)
best_kp1 = np.array(best_kp1)
best_kp2 = np.array(best_kp2)
best_matches = np.array(best_matches)

F, inlier_mask = cv2.findFundamentalMat(best_kp1, best_kp2, cv2.FM_7POINT)
inlier_mask = inlier_mask.flatten()

#points within epipolar lines
inlier_kp1 = best_kp1[inlier_mask == 1]
inlier_kp2 = best_kp2[inlier_mask == 1]

inlier_matches = best_matches[inlier_mask==1]

print(inlier_kp1)
print(inlier_kp2)

img3 = cv2.drawMatches(I1,kp1,I2,kp2, inlier_matches, flags=2, outImg = None)
plt.imshow(img3),plt.show()

thresh = 0

_, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(inlier_kp1), np.float32(inlier_kp2), F, I1gray.shape[::-1], 1)

stereo_L = cv2.warpPerspective(I1, H1, I1gray.shape[::-1])
stereo_R = cv2.warpPerspective(I2, H2, I2gray.shape[::-1])

plt.imshow(stereo_L)
plt.show()
plt.imshow(stereo_R)
plt.show()