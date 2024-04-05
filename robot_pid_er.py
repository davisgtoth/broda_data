#! /usr/bin/env python3

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

class Driver():
    def __init__(self):
        rospy.init_node('robot_pid_er')

        self.bridge = CvBridge()
        rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback)

        self.vel_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)

        self.img = None
        self.num_pixels_above_bottom = 200
        self.line_cutoff = 700

        self.move = Twist()
        self.kp = 10
        self.lin_speed = 0.4
        self.rot_speed = 1.0

        self.bg_sub = cv2.createBackgroundSubtractorMOG2()
        self.reached_crosswalk = False

        self.state = 'init'

    def callback(self, msg):
        self.img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    # returns centre of road outlined by white lines in img 
    # at y pixels above the bottom of the img
    # if ret_sides is True, returns the left and right indices of the road
    def find_road_centre(self, img, y, ret_sides=False):
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

        if ret_sides:
            return left_index, right_index

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

        # if road_centre != -1:
        #     cv2.imshow('camera feed', cv2.circle(img, (road_centre, height - y), 5, (0, 0, 255), -1))
        #     cv2.waitKey(1)

        return road_centre
    
    def get_error(self, img):
        width = img.shape[1]
        road_centre = self.find_road_centre(img, self.num_pixels_above_bottom)
        if road_centre != -1:
            error = ((width // 2) - road_centre) / (width // 2)
        else:
            error = 0
            self.state = 'truck'
        return error
    
    def check_red(self, img):
        height = img.shape[0]
        cropped_img = img[self.line_cutoff:height]
            
        uh_red = 255; us_red = 255; uv_red = 255
        lh_red = 90; ls_red = 50; lv_red = 230
        lower_hsv_red = np.array([lh_red, ls_red, lv_red])
        upper_hsv_red = np.array([uh_red, us_red, uv_red])
        
        hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2HSV)
        red_mask = cv2.inRange(hsv_img, lower_hsv_red, upper_hsv_red)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours.__len__() == 0:
            return False

        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 1000:
            return False
        else:
            return True

    # return true if the pedestrian is on the cross walk
    def check_pedestrian(self, img):
        height = img.shape[0]
        fg_mask = self.bg_sub.apply(img)

        # cv2.imshow('camera feed', fg_mask)
        # cv2.waitKey(1)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours.__len__() == 0:
            return True
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        road_left, road_right = self.find_road_centre(img, height-(y+h-1), ret_sides=True)

        if road_left - 60 < (x + w//2) < road_right + 60:
            return True
        else:
            return False
    
    def start(self):
        print('starting the show (entering road pid state)')
        self.state = 'road'
    
    def run(self):
        while not rospy.is_shutdown():
            if self.img is not None:
                if self.state == 'init':
                    self.start()

                elif self.state == 'road':
                    if self.reached_crosswalk == False and self.check_red(self.img):
                        print('red detected, going to pedo state')
                        self.state = 'pedo'
                    else:
                        error = self.kp * self.get_error(self.img)
                        # print(error)
                        self.move.linear.x = self.lin_speed
                        self.move.angular.z = self.rot_speed * error
                        self.vel_pub.publish(self.move)

                elif self.state == 'pedo':
                    self.reached_crosswalk = True
                    if self.check_pedestrian(self.img):
                        print('pedestrian detected, waiting...')
                    else:
                        print('no pedestrian, going!')
                        self.move.linear.x = 1.2
                        self.move.angular.z = 0
                        self.vel_pub.publish(self.move)
                        rospy.sleep(0.6)
                        print('crossing crosswalk, going back to road pid state')
                        self.state = 'road'
                
                elif self.state == 'truck':
                    print('in truck state')
                    self.move.linear.x = 0
                    self.move.angular.z = 0
                    self.vel_pub.publish(self.move)

                # if error != self.kp * 1000:
                #     move.linear.x = self.lin_speed
                #     move.angular.z = self.rot_speed * error
                #     self.vel_pub.publish(move)
                # else:
                #     move.linear.x = 0
                #     move.angular.z = 0
                #     self.vel_pub.publish(move)
        
        rospy.sleep(0.1)

if __name__ == '__main__':
    try:
        my_driver = Driver()
        rospy.sleep(1)
        my_driver.run()
    except rospy.ROSInterruptException:
        pass
                