import cv2
import matplotlib.pyplot as plt
import imutils
import numpy as np
from skimage import measure

path = "plate.png"
img = cv2.imread(path)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
s = hsv[:, :, 2]
ret, thresh = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # contors find
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(c)
out = img[y:y + h, x:x + w, :].copy()
gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
labels = measure.label(thresh, connectivity=2, background=0)
charCandidates = np.zeros(thresh.shape, dtype="uint8")
labelMask = np.zeros(thresh.shape, dtype="uint8")
for y in range(len(labels)):
    for x in range(len(labels[y])):
        if labels[y][x] > 2:
            labelMask[y][x] = 255
labelMask2 = np.transpose(labelMask)
flagList = []
flag = False
for r in labelMask2:
    if np.count_nonzero(r) != 0:
        flag = True
    else:
        flag = False
    flagList.append(flag)
imList = []
single_ch = []
for i in range(len(flagList)):
    if flagList[i]:
        single_ch.append(labelMask2[i])
    else:
        if single_ch:
            imList.append(single_ch)
            single_ch = []
for i in range(len(imList)):
    imList[i] = np.transpose(imList[i])
for i in imList:
    plt.imshow(i)
    plt.imsave('sing_ch_test.png', i)
