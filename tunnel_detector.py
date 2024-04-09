#! /usr/bin/env python3

import cv2
import numpy as np

# imgs = [cv2.imread(f'tunnel_images_1/img_{i}.jpg') for i in range(8)]

uh_cactus = 66; us_cactus = 255; uv_cactus = 255
lh_cactus = 56; ls_cactus = 86; lv_cactus = 63
lower_hsv_cactus = np.array([lh_cactus, ls_cactus, lv_cactus])
upper_hsv_cactus = np.array([uh_cactus, us_cactus, uv_cactus])

uh_yoda = 68; us_yoda = 255; uv_yoda = 255
lh_yoda = 57; ls_yoda = 96; lv_yoda = 89
lower_hsv_yoda = np.array([lh_yoda, ls_yoda, lv_yoda])
upper_hsv_yoda = np.array([uh_yoda, us_yoda, uv_yoda])

def find_cactus(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv_img, lower_hsv_cactus, upper_hsv_cactus)
    yoda_mask = cv2.inRange(hsv_img, lower_hsv_yoda, upper_hsv_yoda)
    yoda_mask = cv2.bitwise_not(yoda_mask)
    mask = cv2.bitwise_and(green_mask, yoda_mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours.__len__() == 0:
        print('no cactus detected - no contours')
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow('mask', mask)
    cv2.imshow('original image', image)
    cv2.waitKey(0)

# for i in range(8):
#     img = cv2.imread(f'img_{i}.jpg')
#     find_cactus(img)



# for i in range(15):
#     img = cv2.imread(f'tunnel_images_2/img_{i}.jpg')
#     imgs.append(img)

imgs = []
for i in range(15):
    img = cv2.imread(f'img_{i}.jpg')
    imgs.append(img)

# for i in range(8):
#     img = cv2.imread(f'car_images_1/img_{i}.jpg')
#     imgs.append(img)

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

uh_mag = 175; us_mag = 255; uv_mag = 255
lh_mag = 150; ls_mag = 90; lv_mag = 110
lower_hsv_mag = np.array([lh_mag, ls_mag, lv_mag])
upper_hsv_mag = np.array([uh_mag, us_mag, uv_mag])

for image in imgs:
    # gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # skull = cv2.inRange(gray_img, 100, 120)
    # cv2.imshow('gray image', gray_img)
    image_copy = image.copy()
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours.__len__() == 0:
        print('no tunnel detected - no contours')
        cv2.imshow('original image', image)
        cv2.waitKey(0)
        continue
    good_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 30]
    # print(cv2.contourArea(largest_contour))
    if good_contours.__len__() == 0:
        print('no tunnel detected - too small')
        cv2.imshow('original image', image)
        cv2.waitKey(0)
        continue
    combined_contour = np.concatenate(good_contours)
    cv2.drawContours(image, [combined_contour], -1, (0, 255, 0), 3)
    x, y, w, h = cv2.boundingRect(combined_contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    x_mid = x + w // 2
    print(f'x_mid: {x_mid}')
    cv2.line(image, (x_mid, 0), (x_mid, image.shape[0]), (0, 0, 255), 2)
    cv2.line(image, (500, 0), (500, image.shape[0]), (0, 0, 0), 2)
    cv2.line(image, (300, 0), (300, image.shape[0]), (0, 0, 0), 2)
    cv2.imshow('mask', mask)
    cv2.imshow('original image', image)

    hsv_img = cv2.cvtColor(image_copy, cv2.COLOR_BGR2HSV)
    magenta_mask = cv2.inRange(hsv_img, lower_hsv_mag, upper_hsv_mag)

    mag_contours, _ = cv2.findContours(magenta_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if mag_contours.__len__() == 0:
        print('no magenta detected - no contours')
        cv2.imshow('magenta mask', magenta_mask)
        cv2.waitKey(0)
        continue
    largest_mag_contour = max(mag_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_mag_contour)
    cv2.rectangle(image_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # rect = cv2.minAreaRect(largest_mag_contour)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # cv2.drawContours(image_copy, [box], 0, (0, 0, 255), 2)
    cv2.line(image_copy, (x + w // 2, 0), (x + w // 2, image_copy.shape[0]), (0, 0, 255), 2)
    print(f'magenta x mid: {x + w // 2}')
    # cv2.line(image_copy, (int(rect[0][0]), 0), (int(rect[0][0]), image_copy.shape[0]), (0, 0, 255), 2)
    # print(f'magenta x mid: {rect[0][0]}')
    print(f'area: {cv2.contourArea(largest_mag_contour)}')
    print(f'y: {y}, h: {h}')
    cv2.imshow('magenta mask', magenta_mask)
    cv2.imshow('original image copy', image_copy)

    cv2.waitKey(0)