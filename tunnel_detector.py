#! /usr/bin/env python3

import cv2
import numpy as np

imgs = [cv2.imread(f'tunnel_images_1/img_{i}.jpg') for i in range(8)]

for i in range(15):
    img = cv2.imread(f'tunnel_images_2/img_{i}.jpg')
    imgs.append(img)

# height, width = imgs[0].shape[:2]
# imgs = [cv2.resize(img, (width // 3, height // 3)) for img in imgs]

# ho1 = np.hstack((imgs[0], imgs[1], imgs[2], imgs[3]))
# ho2 = np.hstack((imgs[4], imgs[5], imgs[6], imgs[7]))
# img_array = np.vstack((ho1, ho2))
# cv2.imshow('tunnel images', img_array)
# cv2.waitKey(0)

# cv2.imwrite('tunnel_images_1/tunnel_array.jpg', img_array)

uh = 9; us = 255; uv = 255
lh = 0; ls = 106; lv = 66
lower_hsv= np.array([lh, ls, lv])
upper_hsv= np.array([uh, us, uv])

for image in imgs:
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours.__len__() == 0:
        print('no tunnel detected - no contours')
        continue
    good_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
    # print(cv2.contourArea(largest_contour))
    if good_contours.__len__() == 0:
        print('no tunnel detected - too small')
        continue
    combined_contour = np.concatenate(good_contours)
    cv2.drawContours(image, [combined_contour], -1, (0, 255, 0), 3)
    x, y, w, h = cv2.boundingRect(combined_contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow('mask', mask)
    cv2.imshow('original image', image)
    cv2.waitKey(0)