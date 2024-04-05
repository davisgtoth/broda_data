#! /usr/bin/env python3

import cv2
import numpy as np

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

# img_test = cv2.imread('sign_images_3_false/img_30.jpg')
# cv2.imshow('img_test', cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY))
# cv2.waitKey(0)

# Code for cropping each image to the sign
# sign images 1 - 8 images
# sign images 2 - 8 images
# sign images 3 - 44 images
# sign images 4 - 61 images
# sign images 5 - 22 images
for i in range(22):
    img = cv2.imread(f'sign_images_5/img_{i}.jpg')
    # img = cv2.imread(f'sign_images_2/img_{i}.jpg')
    # cv2.imshow('image', img)

    # Threshold the whole sign such that it is black (all blue and white parts)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv) 
    # cv2.imshow('thresholded image', mask)

    # Threshold just the white part of the sign such that it is white
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray img', gray_img)
    sign_mask1 = cv2.inRange(gray_img, 98, 105)
    sign_mask2 = cv2.inRange(gray_img, 197, 205)
    sign_mask3 = cv2.inRange(gray_img, 119, 125)
    sign_mask = cv2.bitwise_or(sign_mask1, sign_mask2)
    sign_mask = cv2.bitwise_or(sign_mask, sign_mask3)
    # cv2.imshow('sign_mask', sign_mask)
    
    # # Draw the contours of both thresholdings
    # contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), cv2.FILLED)
    # cv2.imshow('contours1', img)
    # countours, _ = cv2.findContours(sign_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, countours, -1, (0, 255, 0), cv2.FILLED)
    # cv2.imshow('contours2', img)

    # Combine the two masks
    mask_not = cv2.bitwise_not(mask)
    combined_mask = cv2.bitwise_and(mask_not, sign_mask)
    # cv2.imshow('combined_mask', combined_mask)

    # Find the largest contour and crop the image
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    # img_copy = img.copy()
    # cv2.imshow('rectangle', cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2))
    print(f'----------- image {i} -----------')
    # print(w, h)

    # Roughly find the corners of the sign
    
    # # My code:
    # overlapping_points = []
    # for point in largest_contour:
    #     if point[0][0] <= x+10 or point[0][0] >= x+w-11:
    #         stored_point = (point[0][0], point[0][1])
    #         overlapping_points.append(stored_point)
            # print(stored_point)
    # corner1 = ()
    # corner2 = ()
    # corner3 = ()
    # corner4 = ()
    # ymin1 = 10000
    # ymax1 = 0
    # ymin2 = 10000
    # ymax2 = 0
    # for point in overlapping_points:
    #     if point[0] <= x+10:
    #         if point[1] <= ymin1:
    #             ymin1 = point[1]
    #             corner1 = point
    #         if point[1] >= ymax1:
    #             ymax1 = point[1]
    #             corner3 = point
    #     elif point[0] >= x+w-11:
    #         if point[1] <= ymin2:
    #             ymin2 = point[1]
    #             corner2 = point
    #         if point[1] >= ymax2:
    #             ymax2 = point[1]
    #             corner4 = point

    # # Copilot's more compact code:
    overlapping_points = [(point[0][0], point[0][1]) for point in largest_contour if point[0][0] <= x+10 or point[0][0] >= x+w-11]

    corner1 = min((point for point in overlapping_points if point[0] <= x+10), key=lambda p: p[1], default=())
    corner2 = min((point for point in overlapping_points if point[0] >= x+w-11), key=lambda p: p[1], default=())

    corner3 = max((point for point in overlapping_points if point[0] <= x+10), key=lambda p: p[1], default=())
    corner4 = max((point for point in overlapping_points if point[0] >= x+w-11), key=lambda p: p[1], default=())

    # print(corner1, corner2, corner3, corner4)
    # print('----------- next image -----------')

    img_copy = img.copy()
    cv2.circle(img_copy, corner1, 5, (0, 0, 255), -1)
    cv2.circle(img_copy, corner2, 5, (0, 0, 255), -1)
    cv2.circle(img_copy, corner3, 5, (0, 0, 255), -1)
    cv2.circle(img_copy, corner4, 5, (0, 0, 255), -1)
    cv2.imshow('circle img', img_copy)

    # get the area of the contour and filter on it
    area = cv2.contourArea(largest_contour)
    print(area)
    if area < 6000:
        print('filtered on size')
        cv2.destroyWindow('cropped_img_perspective')
        cv2.waitKey(0)
        continue
    
    # print('sign detected')
    # Crop the image and perspective transform it
    # cropped_img = img[y:y+h, x:x+w]
    # cv2.imshow('cropped_img', cropped_img)
    src_pts = np.array([corner1, corner2, corner3, corner4], dtype=np.float32)
    dst_pts = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    cropped_img2 = cv2.warpPerspective(img, M, (w, h))
    # cv2.imshow('cropped_img_perspective', cropped_img2)
    # final_img = cv2.resize(cropped_img2, (w*4, h*4))
    # cv2.imshow('final_img', final_img)

    # for detecting red
    # uh_crop = 255
    # us_crop = 255
    # uv_crop = 255
    # lh_crop = 170
    # ls_crop = 75
    # lv_crop = 0

    # for detecting blue
    uh_crop = 130
    us_crop = 255
    uv_crop = 255
    lh_crop = 120
    ls_crop = 100
    lv_crop = 50

    lower_hsv_crop = np.array([lh_crop, ls_crop, lv_crop])
    upper_hsv_crop = np.array([uh_crop, us_crop, uv_crop])

    # hsv_cropped_img = cv2.cvtColor(cropped_img2, cv2.COLOR_BGR2HSV) # detect red
    hsv_cropped_img = cv2.cvtColor(cropped_img2, cv2.COLOR_RGB2HSV) # detect blue and flip channels
    mask_cropped = cv2.inRange(hsv_cropped_img, lower_hsv_crop, upper_hsv_crop)

    if np.any(mask_cropped):
        print('sign detected')
        cv2.imshow('cropped_img_perspective', cropped_img2)
    else:
        print('filtered on red circle')
        cv2.destroyWindow('cropped_img_perspective')
        cv2.waitKey(0)
        continue

    cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imwrite(f'sign_images_1/cropped_img_{i}.jpg', cropped_img2)
    # cv2.imwrite(f'sign_images_2/cropped_img_{i}.jpg', cropped_img2)
    # cv2.imwrite(f'sign_images_3_false/cropped_img_{i}.jpg', cropped_img2)
    # cv2.imwrite(f'sign_images_4/cropped_img_{i}.jpg', cropped_img2)
    # cv2.imwrite(f'sign_images_5/cropped_img_{i}.jpg', cropped_img2)


