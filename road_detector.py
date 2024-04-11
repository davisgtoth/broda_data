#! /usr/bin/env python3

import cv2
import numpy as np

# img = cv2.imread('mountain_images_1/img_37.jpg')
# gry_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray image', gry_img)
# cv2.waitKey(0)

# imgs = [cv2.imread(f'road_images_1/img_{i}.jpg') for i in range(21)]

# uh = 255
# us = 20
# uv = 255
# lh = 0
# ls = 0
# lv = 220

# lower_hsv = np.array([lh,ls,lv])
# upper_hsv = np.array([uh,us,uv])

# for image in imgs:
#     height, width = image.shape[:2]
#     image = cv2.resize(image, (int(width/2), int(height/2)))
#     hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)
#     gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     white_mask = cv2.inRange(gray_img, 250, 255)
    # cv2.imshow('white mask ----------- original image ----------- hsv mask', 
    #            np.hstack((cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR), image, 
    #                       cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))))
    # cv2.waitKey(0)

# y - how high above the bottom of the image to look for the road centre
def find_road_centre(img, y):
    height, width = img.shape[:2]
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # white_mask = cv2.inRange(gray_img, 250, 255)

    left_index = -1
    right_index = -1

    for i in range(width):
        if img[height - y, i] == 255 and left_index == -1:
            left_index = i
        elif img[height - y, i] == 255 and left_index != -1:
            right_index = i

    print(f'left index: {left_index}, right index: {right_index}')

    road_centre = -1
    if left_index != -1 and right_index != -1:
        if right_index - left_index > 450:
            print('distance between indices is greater than 450, putting road centre between')
            road_centre = (left_index + right_index) // 2
        # elif left_index < width // 2:
        #     print('distance is not greater than 150 and left index is less than half the width')
        #     road_centre = (left_index + width) // 2
        # else:
        #     print('distance is not greater than 150 and left index is greater than half the width')
        #     road_centre = right_index // 2
        elif right_index > width // 2:
            print('distance is not greater than 450 and right index is greater than half the width')
            road_centre = right_index // 2
        else:
            print('distance is not greater than 450 and right index is less than half the width')
            road_centre = (left_index + width) // 2
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

    


# for image in imgs:
#     print('--- new image ---')
#     print(check_red(image))
#     # height, width = image.shape[:2]
#     # road_centre_1 = find_road_centre(image, 50)
#     # road_centre_2 = find_road_centre(image, 100)
#     # road_centre_3 = find_road_centre(image, 150)
#     # image = cv2.circle(image, (road_centre_1, height-50), 5, (0, 0, 255), -1)
#     # image = cv2.circle(image, (road_centre_2, height-100), 5, (0, 0, 255), -1)
#     # image = cv2.circle(image, (road_centre_3, height-150), 5, (0, 0, 255), -1)
#     # image = cv2.line(image, (width // 2, 0), (width // 2, height), (0, 255, 0), 2)
#     cv2.imshow('image', image)
#     cv2.waitKey(0)

def check_magenta(img):
    height, width = img.shape[:2]

    uh_mag = 175; us_mag = 255; uv_mag = 255
    lh_mag = 150; ls_mag = 90; lv_mag = 110
    lower_hsv_mag = np.array([lh_mag, ls_mag, lv_mag])
    upper_hsv_mag = np.array([uh_mag, us_mag, uv_mag])

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    magenta_mask = cv2.inRange(hsv_img, lower_hsv_mag, upper_hsv_mag)

    cv2.imshow('magenta mask', magenta_mask)
    # cv2.waitKey(0)

    contours, _ = cv2.findContours(magenta_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours.__len__() == 0:
        print('no magenta detected')
        return False
    largest_contour = max(contours, key=cv2.contourArea)
    print(cv2.contourArea(largest_contour))
    # x, y, w, h = cv2.boundingRect(largest_contour)
    # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # if y >= height - 200:
    #     print('magenta detected, going to desert state')
    # else:
    #     print('staying in road state')

    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
    cv2.line(img, (0, height-200), (width, height-200), (0, 255, 0), 2)

    cv2.imshow('image', img)
    cv2.waitKey(0)


# for i in range(5):
#     image = cv2.imread(f'img_{i}.jpg')
#     check_magenta(image)

# imgs = [cv2.imread(f'desert_images_1/img_{i}.jpg') for i in range(14)]
# height, width = imgs[0].shape[:2]
# imgs = [cv2.resize(img, (width // 3, height // 3)) for img in imgs]
# ho1 = np.hstack((imgs[13], imgs[1], imgs[2], imgs[3]))
# ho2 = np.hstack((imgs[4], imgs[5], imgs[6], imgs[7]))
# ho3 = np.hstack((imgs[8], imgs[9], imgs[10], imgs[11]))
# img_array = np.vstack((ho1, ho2, ho3))
# cv2.imshow('road images', img_array)
# cv2.waitKey(0)

# cv2.imwrite('desert_images_1/desert_array.jpg', img_array)

# for image in imgs:
#     # print('--- new image ---')
#     # image = cv2.GaussianBlur(image, (5, 5), 0)
#     gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # cv2.imshow('gray image', gray_img)

#     # adaptive_thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#     # cv2.imshow('adaptive threshold 1', adaptive_thresh)

#     # adaptive_thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4)
#     # cv2.imshow('adaptive threshold 2', adaptive_thresh)

#     adaptive_thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 6)
#     # cv2.imshow('adaptive threshold 3', adaptive_thresh)

#     kernel = np.ones((3, 3), np.uint8)
#     opening = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
#     # cv2.imshow('opening', opening)

#     # edges = cv2.Canny(image, 100, 200)
#     # cv2.imshow('canny edge detection', edges)

#     # white_mask = cv2.inRange(gray_img, 170, 200)
#     # cv2.imshow('white mask', white_mask)
    
#     uh = 37; us = 98; uv = 255
#     lh = 13; ls = 35; lv = 179
#     lower_hsv = np.array([lh, ls, lv])
#     upper_hsv = np.array([uh, us, uv])

#     hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)
#     # cv2.imshow('first mask', mask)

#     # uh2 = 37; us2 = 98; uv2 = 255
#     # lh2 = 13; ls2 = 94; lv2 = 175
#     # lower_hsv2 = np.array([lh2, ls2, lv2])
#     # upper_hsv2 = np.array([uh2, us2, uv2])
#     # mask2 = cv2.inRange(hsv_img, lower_hsv2, upper_hsv2)

#     # mask2 = cv2.bitwise_not(mask2)
#     # mask = cv2.bitwise_and(mask, mask2)

#     # kernel = np.ones((3, 3), np.uint8)
#     # mask = cv2.erode(mask, kernel, iterations=1)
#     # mask = cv2.dilate(mask, kernel, iterations=1)

#     # mask = cv2.bitwise_and(mask, white_mask)

#     contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     # for contour in contours:
#     #     x, y, w, h = cv2.boundingRect(contour)
#     #     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

#     # contours = sorted(contours, key=cv2.arcLength, reverse=True)


#     # contours = sorted(contours, key=lambda contour: cv2.arcLength(contour, True), reverse=True)
#     # largest_contours = contours[:5]
#     min_perimeter = 750  # Replace with your desired minimum perimeter
#     contours = [contour for contour in contours if cv2.arcLength(contour, True) > min_perimeter]
    
#     # good_contours = []
#     # for cnt in contours:
#     #     x, y, w, h = cv2.boundingRect(cnt)
#     #     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
#     #     if h > 125:
#     #         good_contours.append(cnt)

#     good_contours = [cnt for cnt in contours if cv2.boundingRect(cnt)[3] > 125]


#     largest_contours = contours[:2]

#     epsilon = 0.01 * cv2.arcLength(good_contours[0], True)
#     approx = [cv2.approxPolyDP(contour, epsilon, True) for contour in good_contours]

#     # cv2.drawContours(image, approx, -1, (0, 255, 0), 2)
    
#     height, width = image.shape[:2]
#     # cv2.rectangle(image, (0, height-200), (width, height), (0, 255, 0), 2)


#     # cv2.imshow('original image', image)

#     # blank_image = np.zeros((height, width, 3), np.uint8)
#     blank_image = np.zeros_like(image)
#     cv2.fillPoly(blank_image, approx, (255, 255, 255))
#     # cv2.imshow('blank image', cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY))

#     # cv2.imshow('mask', mask)
#     # cv2.waitKey(0)

def thresh_desert(img):
        uh = 37; us = 98; uv = 255
        lh = 13; ls = 35; lv = 179
        lower_hsv1 = np.array([lh, ls, lv])
        upper_hsv = np.array([uh, us, uv])
        lv = 175
        lower_hsv2 = np.array([lh, ls, lv])

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv_img, lower_hsv1, upper_hsv)
        mask2 = cv2.inRange(hsv_img, lower_hsv2, upper_hsv)
        # cv2.imshow('mask1', cv2.resize(mask1, (mask1.shape[1]//2, mask1.shape[0]//2)))
        # cv2.imshow('mask2', cv2.resize(mask2, (mask2.shape[1]//2, mask2.shape[0]//2)))

        uh_road = 37; us_road = 255; uv_road = 255
        lh_road = 0; ls_road = 27; lv_road = 110
        lower_hsv_road = np.array([lh_road, ls_road, lv_road])
        upper_hsv_road = np.array([uh_road, us_road, uv_road])
        road_mask = cv2.inRange(hsv_img, lower_hsv_road, upper_hsv_road)
        cv2.imshow('road mask', cv2.resize(road_mask, (road_mask.shape[1]//2, road_mask.shape[0]//2)))

        # right_most_white = -1
        # for i in range(1, road_mask.shape[1]):

        #     if road_mask[road_mask.shape[0] - 200, road_mask.shape[1] - i] == 255:
        #         right_most_white = i
        #         break
        # road_mask = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)
        
        # cv2.line(road_mask, (0, road_mask.shape[0] - 200), (road_mask.shape[1], road_mask.shape[0] - 200), (255, 0, 0), 2)
        # cv2.circle(road_mask, (road_mask.shape[1]//2, road_mask.shape[0] - 200), 5, (0, 0, 255), -1)
        # cv2.circle(road_mask, (road_mask.shape[1] - right_most_white, road_mask.shape[0] - 200), 5, (0, 0, 255), -1)

        # cv2.imshow('road mask', cv2.resize(road_mask, (road_mask.shape[1]//2, road_mask.shape[0]//2)))

        ylvl = 215

        desert_road_1 = draw_desert_lines(mask1)
        desert_road_2 = draw_desert_lines(mask2)
        cv2.imshow('desert road 1', cv2.resize(desert_road_1, (desert_road_1.shape[1]//2, desert_road_1.shape[0]//2)))
        cv2.imshow('desert road 2', cv2.resize(desert_road_2, (desert_road_2.shape[1]//2, desert_road_2.shape[0]//2)))

        desert_lines_or = cv2.bitwise_or(desert_road_1, desert_road_2)

        # for y, row in enumerate(desert_lines_or):
        #     white_pixels = np.where(row == 255)[0]
        #     if len(white_pixels) > 0:
        #         print('min y pixel:', y)
        #         break

        desert_lines_or = cv2.cvtColor(desert_lines_or, cv2.COLOR_GRAY2BGR)
        road_centre = find_road_centre(cv2.cvtColor(desert_lines_or, cv2.COLOR_BGR2GRAY), ylvl)

        if road_mask[road_mask.shape[0] - 200, road_centre] == 255:
            print('road detected on hill')

        cv2.line(desert_lines_or, (0, desert_lines_or.shape[0] - ylvl), (desert_lines_or.shape[1], desert_lines_or.shape[0] - ylvl), (0, 255, 0), 2)
        cv2.line(desert_lines_or, (desert_lines_or.shape[1]//2, 0), (desert_lines_or.shape[1]//2, desert_lines_or.shape[0]), 
                 (0, 255, 0), 2)
        cv2.circle(desert_lines_or, (road_centre, desert_lines_or.shape[0] - ylvl), 10, (0, 0, 255), -1)

        cv2.imshow('desert lines or', cv2.resize(desert_lines_or, (desert_lines_or.shape[1]//2, desert_lines_or.shape[0]//2)))

        cv2.imshow('original image', cv2.resize(img, (img.shape[1]//2, img.shape[0]//2)))
        cv2.waitKey(0)

        # contours, _ = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # # contours = sorted(contours, key=lambda contour: cv2.arcLength(contour, True), reverse=True)
        # contours = [cnt for cnt in contours if cv2.arcLength(cnt, True) > 750 and cv2.boundingRect(cnt)[3] > 125]

        # if len(contours) == 0:
        #     print('no contours')
        #     return np.zeros_like(img)

        # epsilon = 0.01 * cv2.arcLength(contours[0], True)
        # approx_cnts = [cv2.approxPolyDP(cnt, epsilon, True) for cnt in contours]

        # blank_img = np.zeros_like(img)

        # return cv2.fillPoly(blank_img, approx_cnts, (255, 255, 255))

def draw_desert_lines(img):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours = sorted(contours, key=lambda contour: cv2.arcLength(contour, True), reverse=True)
    contours = [cnt for cnt in contours if cv2.arcLength(cnt, True) > 750 and cv2.boundingRect(cnt)[3] > 125]
    if len(contours) == 0:
        return np.zeros_like(img)
    epsilon = 0.01 * cv2.arcLength(contours[0], True)
    approx_cnts = [cv2.approxPolyDP(cnt, epsilon, True) for cnt in contours]

    blank_img = np.zeros_like(img)

    return cv2.fillPoly(blank_img, approx_cnts, (255, 255, 255))
    

# for image in imgs:
#     lines = thresh_desert(image)
#     lines = cv2.resize(lines, (lines.shape[1]//2, lines.shape[0]//2))
#     image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
#     cv2.imshow('lines and original image', np.hstack((lines, image)))
#     cv2.waitKey(0)

imgs = [cv2.imread(f'img_{i}.jpg') for i in range(19)]

for image in imgs:
    thresh_desert(image)