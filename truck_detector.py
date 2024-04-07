#! /usr/bin/env python3

import cv2
import numpy as np

imgs = [cv2.imread(f'truck_images_1/img_{i}.jpg') for i in range(21)]

height, width = imgs[0].shape[:2]
# imgs = [cv2.resize(img, (width // 2, height // 2)) for img in imgs]

# for i in range(len(imgs)):
#     print(f'img_{i}.jpg')
#     cv2.imshow('truck images', imgs[i])
#     cv2.waitKey(0)

ho1 = np.hstack((imgs[3], imgs[5], imgs[7]))
ho2 = np.hstack((imgs[10], imgs[11], imgs[13]))
img_array = np.vstack((ho1, ho2))
# cv2.imshow('truck images', img_array)
# cv2.waitKey(0)

# cv2.imwrite('truck_images_1/truck_array.jpg', img_array)

bg_sub = cv2.createBackgroundSubtractorMOG2()

for image in imgs:
    print('-------- new image --------')
    image_copy = image.copy()
    fg_mask = bg_sub.apply(image)
    # cv2.imshow('foreground mask', fg_mask)
    # cv2.imshow('original image', image)
    
    # kernel = np.ones((3, 3), np.uint8)
    # fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
    # fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)
    
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # for contour in contours:
    #     x, y, w, h = cv2.boundingRect(contour)
    #     if w * h > 500:
    #         cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    largest_contour = max(contours, key=cv2.contourArea)
    print(f'area: {cv2.contourArea(largest_contour)}')
    x, y, w, h = cv2.boundingRect(largest_contour)
    cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    print(f'middle: {width // 2}, middle of contour: {x + w // 2}')

    middle_x_contour = x + w // 2
    cv2.circle(image_copy, (middle_x_contour, y + h), 5, (0, 0, 255), -1)

    if cv2.contourArea(largest_contour) > 5000:
        if middle_x_contour < (width // 2) - 200:
            print('truck is close but past middle, going right')
        else:
            print('truck is close and on right, waiting...')
    elif cv2.contourArea(largest_contour) > 600 and middle_x_contour < width // 2:
        print('truck on left but kind of close, going right')
    else:
        print('going left')
    
    # cv2.imshow('foreground mask', fg_mask)
    cv2.imshow('original image', image_copy)
    # cv2.imshow('foreground mask and original image', np.hstack((cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR), image_copy)))
    cv2.waitKey(0)