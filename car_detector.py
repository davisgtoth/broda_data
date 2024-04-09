#! /usr/bin/env python3

import cv2
import numpy as np

imgs = [cv2.imread(f'car_images_1/img_{i}.jpg') for i in range(8)]
# height, width = imgs[0].shape[:2]
# imgs = [cv2.resize(img, (width // 2, height // 2)) for img in imgs]

ho1 = np.hstack((imgs[0], imgs[1], imgs[2], imgs[3]))
ho2 = np.hstack((imgs[4], imgs[5], imgs[6], imgs[7]))
img_array = np.vstack((ho1, ho2))

# cv2.imshow('car images', img_array)
# cv2.waitKey(0)

ho3 = np.hstack((imgs[0], imgs[1]))
ho4 = np.hstack((imgs[5], imgs[7]))
img_array2 = np.vstack((ho3, ho4))

# cv2.imshow('car images', img_array2)
# cv2.waitKey(0)

# cv2.imwrite('car_images_1/car_array_1.jpg', img_array2)

uh1 = 255; us1 = 68; uv1 = 68
lh1 = 0; ls1 = 0; lv1 = 33
lower_hsv1 = np.array([lh1, ls1, lv1])
upper_hsv1 = np.array([uh1, us1, uv1])

uh2 = 255; us2 = 255; uv2 = 114
lh2 = 20; ls2 = 20; lv2 = 0
lower_hsv2 = np.array([lh2, ls2, lv2])
upper_hsv2 = np.array([uh2, us2, uv2])

uh_blue = 150; us_blue = 255; uv_blue = 255
lh_blue = 5; ls_blue = 19; lv_blue = 0
lower_hsv = np.array([lh_blue,ls_blue,lv_blue])
upper_hsv = np.array([uh_blue,us_blue,uv_blue])

# image = cv2.imread('car_images_1/img_0.jpg')
for image in imgs:
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # mask1 = cv2.inRange(hsv_img, lower_hsv1, upper_hsv1)
    # mask2 = cv2.inRange(hsv_img, lower_hsv2, upper_hsv2)
    # mask2 = cv2.bitwise_not(mask2)
    # mask = cv2.bitwise_and(mask1, mask2)

    blue_mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)
    # cv2.imshow('blue mask', blue_mask)

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray image', gray_img)
    white_mask = cv2.inRange(gray_img, 98, 105)

    blue_mask_not = cv2.bitwise_not(blue_mask)
    mask = cv2.bitwise_and(white_mask, blue_mask_not)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours.__len__() == 0:
        print('no car detected - no contours')
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('mask', mask)
    cv2.imshow('original image', image)
    cv2.waitKey(0)
