#! /usr/bin/env python3

import cv2
import numpy as np

# This code is kinda all over the place, it includes some stuff that was run to 
# put a whole bunch of images into one big image to test thresholding, as well 
# as the code to crop each image to the sign and perspective transform it.

# 1. Code for combining all images into one

# imgs = [cv2.imread(f'sign_images_1/img_{i}.jpg') for i in range(8)]
# imgs.append(cv2.imread('img_1.jpg'))
# imgs.append(cv2.imread('img_6.jpg'))
# imgs.append(cv2.imread('img_7.jpg'))
# imgs.append(cv2.imread('img_12.jpg'))
# imgs.append(cv2.imread('img_15.jpg'))
# imgs.append(cv2.imread('img_18.jpg'))
# imgs.append(cv2.imread('img_21.jpg'))
# imgs.append(cv2.imread('img_22.jpg'))
imgs = [cv2.imread(f'road_images_1/img_{i}.jpg') for i in range(21)]
# shrink images
width = imgs[0].shape[1] 
height = imgs[0].shape[0] 
imgs = [cv2.resize(img, (int(width/4), int(height/4))) for img in imgs]

# for i in range(len(imgs)):
#     print(f'img_{i}')
#     cv2.imshow('image', imgs[i])
#     cv2.waitKey(0)

# put images all into one
ho1 = np.hstack((imgs[0], imgs[1], imgs[2], imgs[3], imgs[4]))
ho2 = np.hstack((imgs[5], imgs[6], imgs[7], imgs[8], imgs[9]))
ho3 = np.hstack((imgs[10], imgs[11], imgs[12], imgs[13], imgs[14]))
ho4 = np.hstack((imgs[15], imgs[16], imgs[18], imgs[19], imgs[20]))
img_array = np.vstack((ho1, ho2, ho3, ho4))
cv2.imshow('img1', img_array)
cv2.waitKey(0)

# cv2.imwrite('img_array_1.jpg', img_array)
# cv2.imwrite('img_array_2.jpg', img_array)
# cv2.imwrite('road_array.jpg', img_array)

# # v1
# uh = 140
# us = 255
# uv = 205
# lh = 85
# ls = 65
# lv = 0

# # v2
# uh = 114
# us = 255
# uv = 227
# lh = 5
# ls = 19
# lv = 0

# v3
uh = 150
us = 255
uv = 255
lh = 5
ls = 19
lv = 0

# # v4 test
# uh = 130
# us = 255
# uv = 255
# lh = 118
# ls = 0
# lv = 0

lower_hsv = np.array([lh,ls,lv])
upper_hsv = np.array([uh,us,uv])

hsv_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)
# cv2.imshow('thresholded image', mask)

# contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img_array, contours, -1, (0, 255, 0), cv2.FILLED)
# cv2.imshow('contours', img_array)
# cv2.imwrite('all_imgs.jpg', img_array) 

gray_img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray_img_array', gray_img_array)
sign_mask1 = cv2.inRange(gray_img_array, 99, 105)
sign_mask2 = cv2.inRange(gray_img_array, 197, 205)
sign_mask3 = cv2.inRange(gray_img_array, 119, 125)
sign_mask = cv2.bitwise_or(sign_mask1, sign_mask2)
sign_mask = cv2.bitwise_or(sign_mask, sign_mask3)
# cv2.imshow('sign_mask', sign_mask)

uh2 = 142
us2 = 36
uv2 = 221
lh2 = 113
ls2 = 0
lv2 = 99

lower_hsv2 = np.array([lh2,ls2,lv2])
upper_hsv2 = np.array([uh2,us2,uv2])

mask2 = cv2.inRange(hsv_img, lower_hsv2, upper_hsv2)

mask_not = cv2.bitwise_not(mask)
mask_not2 = cv2.bitwise_not(mask2)
# combined_mask = cv2.bitwise_and(mask_not, mask_not2)
# combined_mask = cv2.bitwise_and(mask2, sign_mask)
combined_mask = cv2.bitwise_and(mask_not, sign_mask)
# cv2.imshow('combined_mask', combined_mask)

# cv2.waitKey(0)