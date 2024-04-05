#! /usr/bin/env python3

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

class SignReader():
    def __init__(self):
        rospy.init_node('sign_reader')

        self.bridge = CvBridge()
        rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback)

        self.vel_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)

        self.img = None
        self.min_sign_area = 6000
        
        self.num_pixels_above_bottom = 200
        self.kp = 5
        self.lin_speed = 0.2
        self.rot_speed = 1.0
        self.no_lines_error = 1000
        
        self.sign_img = None

        self.firstSignTime = None
        self.durationBetweenSigns = rospy.Duration.from_sec(5)

    # callback function for robot camera feed 
    def callback(self, msg):
        self.img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
    def check_if_sign(self, img):
        """
        Checks if a sign is present in the provided image.

        Args:
            img (numpy.ndarray): The image in which to check for a sign.

        Returns:
            numpy.ndarray or None: The cropped image of the sign if found, None otherwise.
        """

        # threshold camera image for blue
        uh_blue = 150; us_blue = 255; uv_blue = 255
        lh_blue = 5; ls_blue = 19; lv_blue = 0
        lower_hsv = np.array([lh_blue,ls_blue,lv_blue])
        upper_hsv = np.array([uh_blue,us_blue,uv_blue])
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)

        # threshold camera image for white
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        white_mask1 = cv2.inRange(gray_img, 98, 105)
        white_mask2 = cv2.inRange(gray_img, 197, 205)
        white_mask3 = cv2.inRange(gray_img, 119, 125)
        white_mask = cv2.bitwise_or(white_mask1, white_mask2)
        white_mask = cv2.bitwise_or(white_mask, white_mask3)

        # combine masks
        blue_mask_not = cv2.bitwise_not(blue_mask)
        combined_mask = cv2.bitwise_and(blue_mask_not, white_mask)

        # find largest contour in the combined mask image
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours.__len__() == 0: # return None if no contours are found
            # print('no sign detected - no contours')
            return None
        largest_contour = max(contours, key=cv2.contourArea)

        # filter out contours that are too small
        area = cv2.contourArea(largest_contour)
        if area < self.min_sign_area:
            # print('no sign detected - too small')
            return None

        # find the corners of the sign
        x, y, w, h = cv2.boundingRect(largest_contour)
        overlapping_points = [(point[0][0], point[0][1]) for point in largest_contour if point[0][0] <= x+10 or point[0][0] >= x+w-11]
        corner1 = min((point for point in overlapping_points if point[0] <= x+10), key=lambda p: p[1], default=())
        corner2 = min((point for point in overlapping_points if point[0] >= x+w-11), key=lambda p: p[1], default=())
        corner3 = max((point for point in overlapping_points if point[0] <= x+10), key=lambda p: p[1], default=())
        corner4 = max((point for point in overlapping_points if point[0] >= x+w-11), key=lambda p: p[1], default=())

        # perspective transform and crop the image
        src_pts = np.array([corner1, corner2, corner3, corner4], dtype='float32')
        dst_pts = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype='float32')
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        cropped_img = cv2.warpPerspective(img, matrix, (w, h))

        # threshold for red in the cropped image
        uh_red = 130; us_red = 255; uv_red = 255
        lh_red = 120; ls_red = 100; lv_red = 50
        lower_hsv_red = np.array([lh_red, ls_red, lv_red])
        upper_hsv_red = np.array([uh_red, us_red, uv_red])
        hsv_cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2HSV)
        red_mask_cropped = cv2.inRange(hsv_cropped_img, lower_hsv_red, upper_hsv_red)

        # filter out if no red in the cropped image
        if not np.any(red_mask_cropped):
            # print('no sign detected - no red')
            return None
        
        # TODO: add code to see if sign is cut off 

        # found a sign !!
        # print('sign detected')
        return cropped_img
    
    def compare_sign(self, new_sign):
        """
        Compares the new sign image to the currently stored sign image. If the new image is 
        larger, it replaces the stored image.

        Args:
            new_sign (numpy.ndarray): The new sign image to compare with the currently stored sign image.

        Returns:
            None
        """
        if self.sign_img is None: # if stored sign image hasn't been assigned yet, assign it
            self.sign_img = new_sign
            self.firstSignTime = rospy.Time.now() # start timer for reading sign
            print('assigned sign image and timer started')
        else:
            if new_sign.size > self.sign_img.size: # compare size of new sign to stored sign
                self.sign_img = new_sign
                print('bigger sign found')
        return

    # when enough time has elapsed from initial sign detection, get the letters from the best 
    # sign image and send them to the neural network
    def read_sign(self):
        # TODO: crop sign to letters and send to NN
        print('sent sign image to NN')
        return
    
    def find_road_centre(self, img, y):
        """
        This method finds the centre of the road in an image.

        Args:
            img (numpy.ndarray): The input image, in BGR format.
            y (int): The y-coordinate from the bottom of the image to find the road centre.

        Returns:
            road_centre (int): The x-coordinate of the road centre, -1 if no road lines are detected.
        """
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

        road_centre = -1
        if left_index != -1 and right_index != -1:
            if right_index - left_index > 150:
                road_centre = (left_index + right_index) // 2
            elif left_index < width // 2:
                road_centre = (left_index + width) // 2
            else:
                road_centre = right_index // 2
        else:
            print('no road lines detected')
            road_centre = -1

        if road_centre != -1:
            cv2.imshow('camera feed', cv2.circle(img, (road_centre, height - y), 5, (0, 0, 255), -1))
            cv2.waitKey(1)

        return road_centre
    
    def get_error(self, img):
        """
        This method calculates the error between the centre of the road and the centre of the image.

        Args:
            img (numpy.ndarray): The input image, in BGR format.

        Returns:
            error (int): The x coordinate difference between the road centre and the centre of the image, 1000 if no road lines are detected.
        """
        width = img.shape[1]
        road_centre = self.find_road_centre(img, self.num_pixels_above_bottom)
        if road_centre != -1:
            error = ((width // 2) - road_centre) / (width // 2)
        else:
            error = self.no_lines_error
        return error

    def run(self):
        while not rospy.is_shutdown():
            # check if robot camera feed sees a sign
            if self.img is not None:
                cropped_img = self.check_if_sign(self.img) # returns None if no sign detected
                if cropped_img is not None:
                    self.compare_sign(cropped_img) # changes self.sign_img if new sign is larger

                error = self.kp * self.get_error(self.img)
                move = Twist()
                if error != self.kp * self.no_lines_error:
                    move.linear.x = self.lin_speed
                    move.angular.z = self.rot_speed * error
                    self.vel_pub.publish(move)
                else:
                    move.linear.x = 0
                    move.angular.z = 0
                    self.vel_pub.publish(move)

            # if we've found a sign ...
            if self.sign_img is not None:
                # display the sign image
                cv2.imshow('sign', self.sign_img)
                cv2.waitKey(1)
                # check if enough time has elapsed to read the sign
                current_time = rospy.Time.now()
                elapsed_time = current_time - self.firstSignTime
                if elapsed_time > self.durationBetweenSigns:
                    self.read_sign()
                    self.sign_img = None
                    self.firstSignTime = None

            rospy.sleep(0.1) # 100ms delay

if __name__ == '__main__':
    try:
        my_bot = SignReader()
        rospy.sleep(1)
        my_bot.run()
    except rospy.ROSInterruptException:
        pass
