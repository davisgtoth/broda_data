#! /usr/bin/env python3

import cv2
import numpy as np

# get cropped images from set 1
imgs = [cv2.imread(f'sign_images_1/cropped_img_{i}.jpg') for i in range(8)]

# get cropped images from set 2
for i in range(8):
    img = cv2.imread(f'sign_images_2/cropped_img_{i}.jpg')
    if img is not None:
        imgs.append(img)

# get cropped images from set 3
for i in range(41):
    try:
        img = cv2.imread(f'sign_images_3/cropped_img_{i}.jpg')
        if img is not None:
            imgs.append(img)
    except Exception as e:
        print(f"An error occurred while reading file cropped_img_{i}.jpg: {e}")

# get cropped images from set 4
for i in range(61):
    try:
        img = cv2.imread(f'sign_images_4/cropped_img_{i}.jpg')
        if img is not None:
            imgs.append(img)
    except Exception as e:
        print(f"An error occurred while reading file cropped_img_{i}.jpg: {e}")

# get cropped images from set 5
for i in range(22):
    try:
        img = cv2.imread(f'sign_images_5/cropped_img_{i}.jpg')
        if img is not None:
            imgs.append(img)
    except Exception as e:
        print(f"An error occurred while reading file cropped_img_{i}.jpg: {e}")

imgs_copy = imgs.copy()

for i in range(len(imgs)):
    # print(imgs[i].shape)
    imgs[i] = cv2.resize(imgs[i], (200, 150))
    # cv2.imshow('image', imgs[i])
    # print(f"image {i}")
    # cv2.waitKey(0)

ho1 = np.hstack((imgs[0], imgs[1], imgs[2], imgs[3]))
ho2 = np.hstack((imgs[4], imgs[5], imgs[6], imgs[7]))
ho3 = np.hstack((imgs[16], imgs[18], imgs[19], imgs[19]))
ho4 = np.hstack((imgs[29], imgs[34], imgs[39], imgs[43]))
ho5 = np.hstack((imgs[50], imgs[51], imgs[55], imgs[40]))
# ho2 = np.hstack((imgs[4], imgs[16], imgs[18], imgs[19]))
# ho3 = np.hstack((imgs[29], imgs[30], imgs[31], imgs[34]))
# ho4 = np.hstack((imgs[39], imgs[40], imgs[43], imgs[44]))
# ho5 = np.hstack((imgs[50], imgs[51], imgs[55], imgs[77]))
img_array = np.vstack((ho1, ho2, ho3, ho4, ho5))
# cv2.imshow('image array', img_array)
cv2.imwrite('sign_array.jpg', img_array)
gray_img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray image array', gray_img_array)
# cv2.waitKey(0)

# can filter false signs based on if the gray image has any pixels of value less than 80
# for i in range(len(imgs)):
#     gray_img = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)
#     if np.any(gray_img < 80):
#         print(f"image {i} is a true sign")
#     else:
#         print(f"image {i} is a false sign")
#     cv2.imshow('image', imgs[i])
#     cv2.waitKey(0)


# can filter false signs if they have the red circle or not with this hsv range
uh = 255
us = 255
uv = 255
lh = 170
ls = 75
lv = 0
lower_hsv = np.array([lh,ls,lv])
upper_hsv = np.array([uh,us,uv])

# hsv_array_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
# mask = cv2.inRange(hsv_array_img, lower_hsv, upper_hsv)
# cv2.imshow('thresholded image', mask)
# cv2.waitKey(0)

for i in range(len(imgs_copy)):
    hsv_img = cv2.cvtColor(imgs_copy[i], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)
    if np.any(mask):
        print(f"image {i} is a true sign")
    else:
        print(f"image {i} is a false sign")
    cv2.imshow('image', imgs_copy[i])
    cv2.waitKey(0)