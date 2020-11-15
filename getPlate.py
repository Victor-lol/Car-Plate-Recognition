#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 19:33:09 2020

@author: victor
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2 
import os

path = "images/Cars1.png"
img1 = cv2.imread(path) 
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img = cv2.GaussianBlur(img1, (3, 3), 0)
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, img_thresh = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

edge = cv2.Canny(img_thresh,100,200)

kernel = np.ones((5,19),np.uint8)
close_img = cv2.morphologyEx(edge, cv2.MORPH_CLOSE,kernel)
open_img = cv2.morphologyEx(close_img, cv2.MORPH_OPEN,kernel)

kernel = np.ones((11,5),np.uint8)
open_img = cv2.morphologyEx(open_img, cv2.MORPH_OPEN,kernel)

_,cnts,_= cv2.findContours(open_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


block = []

for c in cnts:
    y,x=[],[]  
    for p in c:  
        y.append(p[0][0])  
        x.append(p[0][1])  
    rect = cv2.minAreaRect(c)
    h,w = rect[1]

    A = h * w
    if h > w:
        w, h = h, w
    r =  w / h
    if  r > 2 and r < 9.5:
        block.append([[[min(y), min(x)], [max(y), max(x)]]])
block=sorted(block,key=lambda b: b[0][1])

maxWeight, maxIndex = 0, -1
# print(block, end=' ')

for i in range(len(block)):
   
    start =  block[i][0][0]
    end = block[i][0][1]
    candidate = img[start[1]:end[1], start[0]:end[0]]
    
    hsv = cv2.cvtColor(candidate,cv2.COLOR_BGR2HSV)
    lower=np.array([0, 0,0])  
    upper=np.array([255,255,255])
    mask = cv2.inRange(hsv, lower, upper)
    
    w1 = 0
    for m in mask:
        w1 += m / 255
    w2 = 0
    for wi in w1:
        w2 += wi
    
    if w2 > maxWeight:
        maxIndex = i
        maxWeight = w2

loc = block[maxIndex]
# print(loc)
start = (loc[0][0][0], loc[0][0][1])
end = (loc[0][1][0], loc[0][1][1])
# print(start, end)
    

draw = cv2.rectangle(img,start, end,(0,255,0), 2)
plt.imshow(draw)
plt.imshow(img, cmap='gray')

# plt.imsave('plate.png',img[start[1]:end[1],start[0]:end[0],:])




