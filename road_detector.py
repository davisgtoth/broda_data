#! /usr/bin/env python3

import cv2
import numpy as np

imgs = [cv2.imread(f'road_images_1/img_{i}.jpg') for i in range(21)]

uh = 255
us = 20
uv = 255
lh = 0
ls = 0
lv = 220

lower_hsv = np.array([lh,ls,lv])
upper_hsv = np.array([uh,us,uv])

for image in imgs:
    height, width = image.shape[:2]
    image = cv2.resize(image, (int(width/2), int(height/2)))
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    white_mask = cv2.inRange(gray_img, 250, 255)
    # cv2.imshow('white mask ----------- original image ----------- hsv mask', 
    #            np.hstack((cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR), image, 
    #                       cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))))
    # cv2.waitKey(0)

# y - how high above the bottom of the image to look for the road centre
def find_road_centre(img, y):
    height, width = img.shape[:2]
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    white_mask = cv2.inRange(gray_img, 250, 255)

    left_index = -1
    right_index = -1

    for i in range(width):
        if white_mask[height - y, i] == 255 and left_index == -1:
            left_index = i
        elif white_mask[height - y, i] == 255 and left_index != -1:
            right_index = i

    print(f'left index: {left_index}, right index: {right_index}')

    road_centre = -1
    if left_index != -1 and right_index != -1:
        if right_index - left_index > 150:
            print('distance between indices is greater than 150')
            road_centre = (left_index + right_index) // 2
        elif left_index < width // 2:
            print('distance is not greater than 150 and left index is less than half the width')
            road_centre = (left_index + width) // 2
        else:
            print('distance is not greater than 150 and left index is greater than half the width')
            road_centre = right_index // 2
    else:
        print('indices not assigned')
        road_centre = width // 2

    return road_centre    

def check_red(img):
    height, width = img.shape[:2]
    y_cutoff = 500
    cropped_img = img[y_cutoff:height]

    cv2.imshow('cropped image', cropped_img)
        
    uh_red = 255; us_red = 255; uv_red = 255
    lh_red = 90; ls_red = 50; lv_red = 230
    lower_hsv_red = np.array([lh_red, ls_red, lv_red])
    upper_hsv_red = np.array([uh_red, us_red, uv_red])
    
    hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2HSV)
    red_mask = cv2.inRange(hsv_img, lower_hsv_red, upper_hsv_red)
    cv2.imshow('masked image', red_mask)
    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours.__len__() == 0:
        return False

    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 1000:
        return False
    else:
        return True

    


for image in imgs:
    print('--- new image ---')
    print(check_red(image))
    # height, width = image.shape[:2]
    # road_centre_1 = find_road_centre(image, 50)
    # road_centre_2 = find_road_centre(image, 100)
    # road_centre_3 = find_road_centre(image, 150)
    # image = cv2.circle(image, (road_centre_1, height-50), 5, (0, 0, 255), -1)
    # image = cv2.circle(image, (road_centre_2, height-100), 5, (0, 0, 255), -1)
    # image = cv2.circle(image, (road_centre_3, height-150), 5, (0, 0, 255), -1)
    # image = cv2.line(image, (width // 2, 0), (width // 2, height), (0, 255, 0), 2)
    cv2.imshow('image', image)
    cv2.waitKey(0)