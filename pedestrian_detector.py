#! /usr/bin/env python3

import cv2
import numpy as np

imgs = [cv2.imread(f'pedestrian_images_2/img_{i}.jpg') for i in range(46)]

imgs2 = [cv2.imread(f'pedestrian_images_3/img_{i}.jpg') for i in range(33)]

# for image in imgs:
#     cv2.imshow('pedestrian images', image)
#     cropped_img = image[320:440, 400:920]
#     cv2.imshow('cropped pedestrian images', cropped_img)
#     cv2.waitKey(0)

# for image in imgs2:
#     cv2.imshow('pedestrian images', image)
#     cropped_img = image[320:440, 400:920]
#     cv2.imshow('cropped pedestrian images', cropped_img)
#     cv2.waitKey(0)

# height, width = imgs[0].shape[:2]
imgs = [img[320:440, 400:920] for img in imgs]

# ho1 = np.hstack((imgs[0], imgs[1], imgs[2], imgs[3]))
# ho2 = np.hstack((imgs[4], imgs[5], imgs[6], imgs[7]))
# ho1 = np.hstack((imgs[0], imgs[1]))
# ho2 = np.hstack((imgs[2], imgs[4]))
# img_array = np.vstack((ho1, ho2))
# cv2.imshow('pedestrian images', img_array)
# cv2.waitKey(0)

# cv2.imwrite('pedestrian_images_1/pedestrian_array_1.jpg', img_array)

def find_road_edges(img, y):
    height, width = img.shape[:2]
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    white_mask = cv2.inRange(gray_img, 250, 255)
    
    left_index = -1
    right_index = -1

    # cv2.imshow('white mask', white_mask)

    for i in range(width):
        if white_mask[y, i] == 255 and left_index == -1:
            left_index = i
        elif white_mask[y, i] == 255 and left_index != -1:
            right_index = i
    # print(left_index, right_index)
    return left_index, right_index

bg_sub = cv2.createBackgroundSubtractorMOG2()

for image in imgs:
    image_copy = image.copy()
    fg_mask = bg_sub.apply(image)
    # cv2.imshow('foreground mask', fg_mask)
    # cv2.imshow('original image', image)
    
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print('could not find pedestrian')
        continue
    largest_contour = max(contours, key=cv2.contourArea)
    print(f'area: {cv2.contourArea(largest_contour)}')
    x, y, w, h = cv2.boundingRect(largest_contour)
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.circle(image, (x+w//2,y+h), 5, (0, 0, 255), -1)

    road_left, road_right = find_road_edges(image_copy, y+h-1)
    # print(x+(w//2))
    cv2.circle(image, (road_left, y+h), 5, (0, 0, 255), -1)
    cv2.circle(image, (road_right, y+h), 5, (0, 0, 255), -1)
    if road_left - 60 < (x + w//2) < road_right + 80:
        print('pedestrian on road WAIT')
    else:
        print('no ped on road, GOOOO')

    img_array = np.hstack((image, cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)))
    cv2.imshow('foreground mask and original image', img_array)
    cv2.waitKey(0)

red_line_images = [cv2.imread(f'crosswalk_images_1/img_{i}.jpg') for i in range(30)]
for image in red_line_images:
    # print('------- new image -------')
    uh_red = 255; us_red = 255; uv_red = 255
    lh_red = 90; ls_red = 50; lv_red = 230
    lower_hsv_red = np.array([lh_red, ls_red, lv_red])
    upper_hsv_red = np.array([uh_red, us_red, uv_red])
    
    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    red_mask = cv2.inRange(hsv_img, lower_hsv_red, upper_hsv_red)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    # print(f'area: {cv2.contourArea(largest_contour)}')

    rect = cv2.minAreaRect(largest_contour)
    # print(f'x, y: {round(rect[0][0], 1)}, {round(rect[0][1], 1)}')
    # print(f'width, height: {round(rect[1][0], 1)}, {round(rect[1][1], 1)}')
    # print(f'angle: {round(rect[2], 1)}')
    
    height, width = image.shape[:2]
    image = cv2.resize(image, (int(width/2), int(height/2)))
    red_mask = cv2.resize(red_mask, (int(width/2), int(height/2)))
    display_img = np.hstack((image, cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)))

    # print()

    # if cv2.contourArea(largest_contour) < 1800:
    #     print('red line too small')
    # elif rect[0][1] < 475:
    #     print('red line too far away')
    # elif 1 < rect[2] < 89:
    #     print('red line too angled')
    # else:
    #     print('red line good, going to ped state')

    # print()

    # cv2.imshow('red line images', display_img)
    # cv2.waitKey(0)