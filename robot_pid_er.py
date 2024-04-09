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

        self.state = 'init' # init, road, ped, truck, desert, yoda, tunnel, mountain
        
        # image variables
        self.img = None
        self.img_height = 0
        self.img_width = 0
        
        self.cycle_count = 0

        # PID controller variables
        self.move = Twist()
        self.lin_speed = 0.5 # defualt PID linear speed of robot
        self.rot_speed = 1.0 # base PID angular speed of robot

        self.road_line_width = 150
        self.road_min_white_val = 250
        self.road_max_white_val = 255
        
        self.kp = 11 # proportional gain for PID controller
        self.road_buffer = 200 # pixels above bottom of image to find road centre
        self.speed_buffer = 1.3 # buffer for gradual speed increase/decrease

        self.accel_rate = 0.1 # velocity to increase by with each loop
        self.decel_rate = 0.1 # velocity to decrease by with each loop
        self.accel_freq = 50 # frequency of loop when increasing/decreasing speed
        
        # Pedestraian detection variables
        self.reached_crosswalk = False
        self.red_line_min_area = 1000 # minimum contour area for red line
        self.red_line_approach_lin_vel = 0.4
        self.red_line_approach_rot_vel = 0.3
        self.red_line_min_angle = 1.0
        self.red_line_max_angle = 89.0
        self.red_line_stop_y = 400
       
        self.bg_sub = cv2.createBackgroundSubtractorMOG2()
        self.ped_crop_x_min = 400 # values for cropping image to crosswalk and pedestrian
        self.ped_crop_x_max = 920
        self.ped_crop_y_min = 320
        self.ped_crop_y_max = 440
        
        self.ped_left_buffer = 60 # lateral pixel buffers for pedestrian detection
        self.ped_right_buffer = 80
        self.ped_min_area = 400 # minimum contour area for detecting the pedestrian
                
        self.ped_safe_count = 0
        self.ped_safe_count_buffer = 5
        
        self.ped_lin_speed = 2.5 # linear speed of robot when crossing crosswalk
        self.ped_ang_speed = 0 # angular speed of robot when crossing crosswalk
        self.ped_sleep_time = 0.01 # time to sleep when crossing crosswalk

        # Truck detection variables
        self.reached_truck = False
        self.truck_init_cycle = 0
        self.truck_cycle_buffer = 15

        self.truck_left_area = 580 # goes right if truck area is bigger than this
        self.truck_wait_area = 7000
        
        self.truck_min_area = 5000 # used in function, but never actually called, check later
        
        self.truck_turn_dir = ''
        self.truck_left_turn_amplifier = 1.5 # amplifies error when find no road to turn left
        self.truck_right_kp = 11
        self.truck_right_lin_speed = 0.5 #0.7

        self.truck_to_desert_sleep = 0.2 # time to go straight when transitioning from truck to desert states

        # Desert detection variables
        self.desert_min_arc_length = 750 
        self.desert_line_cnt_min_height = 125
        
        self.desert_min_magenta_area = 8000
        self.desert_past_magenta_line_area = 1000
        
        self.desert_road_buffer = 250

        self.magneta_min_angle = 0.5
        self.magneta_max_angle = 89.5
        self.magenta_angle_lin_speed = 0.4
        self.magenta_angle_rot_speed = 0.3

        # Yoda detection variables
        self.reached_yoda = False
        
        self.cactus_min_area = 630
        self.cactus_max_area = 800
        self.cactus_lin_speed = 0.6
        
        self.tunnel_turn_speed = 4.0
        self.tunnel_min_area = 30
        self.tunnel_mid_x = 500
        
        self.over_hill = False
        self.hill_lin_speed = 0.6
        self.hill_rot_speed = -0.2

        self.yoda_mag_min_area_for_y = 5000 # get the y value of the magenta line only when the contour area is greater than this
        self.yoda_find_mag_min_area = 100 # minimum contour area of the magenta line to say that it detects the line
        self.yoda_mag_x_mid = 500 # x coord of the magenta line to pid to
        self.yoda_mag_y_exit = 650 # starts going to tunnel state when y value of mageneta line is greater than this

        # Tunnel detection variables
        self.tunnel_pid_height = 150

    # callback function for camera subscriber
    def callback(self, msg):
        self.img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.img_height, self.img_width = self.img.shape[:2]
        self.cycle_count += 1

    def find_road_centre(self, img, y, width, height, ret_sides=False):
        """
        Returns the centre of the road at y pixels above the bottom of the image in a thresholded 
        image where the road is outlined in white. If ret_sides is True, returns the left and 
        right indices of the road.

        Parameters:
        img (numpy.ndarray): Thresholded image where the road is outlined in white (255).
        y (int): The number of pixels above the bottom of the image to find the centre.
        ret_sides (bool, optional): If set to True, the function returns the left and right 
                                    indices of the road. Default is False.

        Returns:
        int or tuple: If ret_sides is True, the function returns a tuple (left_index, right_index) 
                        representing the left and right indices of the road.
                      If ret_sides is False, the function returns an integer representing the center
                        of the road. If the road center cannot be determined, the function returns -1.
        """
        left_index = right_index = -1
        for i in range(width):
            if img[height - y, i] == 255 and left_index == -1:
                left_index = i
            elif img[height - y, i] == 255 and left_index != -1:
                right_index = i

        if ret_sides:
            return left_index, right_index

        road_centre = -1
        if left_index != -1 and right_index != -1:
            if right_index - left_index > self.road_line_width:
                road_centre = (left_index + right_index) // 2
            elif left_index < width // 2:
                road_centre = (left_index + width) // 2
            else:
                road_centre = right_index // 2
        else:
            road_centre = -1

        # if road_centre != -1:
        #     cv2.imshow('camera feed', cv2.circle(img, (road_centre, height - y), 5, (0, 0, 255), -1))
        #     cv2.waitKey(1)

        return road_centre
    
    # returns the error between the centre of the road and the centre of a thresholded image
    # for either a road or desert image, default is road
    # returns error of 0 if no road lines are found on either side
    # enters the truck state if no road is detected and have reached the crosswalk
    def get_error(self, img):
        """
        Returns the error between the centre of the road and the centre of a thresholded image
        for either a road or desert image, default is road. Returns error of 0 if no road lines 
        are found on either side. Enters the truck state if no road is detected and have reached 
        the crosswalk.

        Parameters:
        img (numpy.ndarray): The input image.
        road (bool, optional): If True, the function processes the image as a road image. Default is True.
        desert (bool, optional): If True, the function processes the image as a desert image. Default is False.

        Returns:
        float: The error between the centre of the road and the centre of the image. Returns 0 if no 
        road lines are found on either side.
        """
        if self.state == 'road' or self.state == 'truck':
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = cv2.inRange(gray_img, self.road_min_white_val, self.road_max_white_val)
        elif self.state == 'desert':
            mask = cv2.cvtColor(self.thresh_desert(img), cv2.COLOR_BGR2GRAY)
            self.road_buffer = self.desert_road_buffer
            # cv2.imshow('desert mask', cv2.resize(mask, (self.img_width // 2, self.img_height // 2)))
            # cv2.waitKey(1)
        elif self.state == 'tunnel':
            mask = self.find_tunnel(img, ret_mask=True)
            self.road_buffer = self.tunnel_pid_height

        road_centre = self.find_road_centre(mask, self.road_buffer, self.img_width, self.img_height)
        if road_centre != -1:
            error = ((self.img_width // 2) - road_centre) / (self.img_width // 2)
        elif self.reached_crosswalk and not self.reached_truck:
            error = 0
            print('no road detected, going to truck state')
            self.state = 'truck'
            self.truck_init_cycle = self.cycle_count
        elif self.truck_turn_dir == 'left':
            error = self.truck_left_turn_amplifier * ((self.img_width // 2) - (self.img_width // 4)) / (self.img_width // 2)
        elif self.truck_turn_dir == 'right':
            error = ((self.img_width // 2) - (3 * self.img_width // 4)) / (self.img_width // 2)
        else:
            error = 0
        return error
    
    def check_red(self, img, ret_angle=False, ret_y=False):
        """
        Checks if red is found in the image with an area greater than red_line_min_area.

        Parameters:
        img (numpy.ndarray): The input image.

        Returns:
        bool: Returns True if red is found in the image with an area greater than red_line_min_area, otherwise False.
        """
        uh_red = 255; us_red = 255; uv_red = 255
        lh_red = 90; ls_red = 50; lv_red = 230
        lower_hsv_red = np.array([lh_red, ls_red, lv_red])
        upper_hsv_red = np.array([uh_red, us_red, uv_red])
        
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        red_mask = cv2.inRange(hsv_img, lower_hsv_red, upper_hsv_red)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours.__len__() == 0:
            return False

        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)

        if not ret_angle and not ret_y:
            if cv2.contourArea(largest_contour) < self.red_line_min_area:
                return False
            else:
                return True
        elif ret_angle:
            return rect[2]
        elif ret_y:
            return rect[0][1]

    # return true if the pedestrian is on the cross walk or within the 
    def check_pedestrian(self, img):
        """
        Checks if a pedestrian is on the crosswalk or within a buffer distance to the road.

        Parameters:
        img (numpy.ndarray): The input image.

        Returns:
        bool: Returns True if a pedestrian is detected on the crosswalk or within the buffer, otherwise False.
        """
        cropped_img = img[self.ped_crop_y_min:self.ped_crop_y_max, self.ped_crop_x_min:self.ped_crop_x_max]
        height, width = cropped_img.shape[:2]
        fg_mask = self.bg_sub.apply(cropped_img)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours.__len__() == 0:
            return False
        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) < self.ped_min_area:
            return False

        x, y, w, h = cv2.boundingRect(largest_contour)

        gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        white_mask = cv2.inRange(gray_img, self.road_min_white_val, self.road_max_white_val)
        ped_height_from_bottom = height - (y + h - 1)
        road_left, road_right = self.find_road_centre(white_mask, ped_height_from_bottom, width, height, ret_sides=True)

        if road_left == -1 and road_right == -1:
            return True

        # cv2.rectangle(cropped_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # cv2.circle(cropped_img, (x + w//2, y + h), 5, (0, 0, 255), -1)
        # cv2.circle(cropped_img, (road_left, y+h), 5, (0, 0, 255), -1)
        # cv2.circle(cropped_img, (road_right, y+h), 5, (0, 0, 255), -1)
        
        # cv2.imshow('camera feed', np.hstack((cropped_img, cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR))))
        # cv2.waitKey(1)
        
        if road_left - self.ped_left_buffer < (x + w//2) < road_right + self.ped_right_buffer:
            return True
        else:
            return False
        
    def drive_robot(self, linear, angular):
        """
        Publishes a velocity command. If the linear velocity to be published is greater than or
        less than the current linear velocity by more/less than 1.5, the function gradually
        increases/decreases the linear velocity. 

        Parameters:
        linear (float): The linear velocity in the x direction.
        angular (float): The angular velocity in the z direction.

        Returns:
        None
        """
        if linear >  self.move.linear.x + self.speed_buffer:
            rate = rospy.Rate(self.accel_freq)
            vel = self.move.linear.x
            while vel < linear:
                self.move.linear.x = vel
                self.move.angular.z = 0
                self.vel_pub.publish(self.move)
                vel += self.accel_rate
                rate.sleep()

        elif linear < self.move.linear.x - self.speed_buffer:
            vel = self.move.linear.x
            rate = rospy.Rate(self.accel_freq)
            while vel > linear:
                self.move.linear.x = vel
                self.move.angular.z = 0
                self.vel_pub.publish(self.move)
                vel -= self.decel_rate
                rate.sleep()
        else:
            self.move.linear.x  = linear
            self.move.angular.z = angular
            self.vel_pub.publish(self.move)

    # returns true if it detects that the truck is big, if at intersection, returns contour area and mid x point
    def check_truck(self, img, at_intersection=False):
        fg_mask = self.bg_sub.apply(img)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours.__len__() == 0:
            return 0, 0 if at_intersection else True
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # cv2.imshow('fg mask', fg_mask)
        # cv2.waitKey(1)

        if at_intersection:
            return cv2.contourArea(largest_contour), x + w // 2
        
        # TODO: check if this is used, don't think it is
        elif cv2.contourArea(largest_contour) > self.truck_min_area:
            print('testing code not used for truck, here')
            return True
        else:
            print('also testing code not used for truck, here')
            return False
    
    # returns true if there is magenta at or below the point where we detect for road lines
    def check_magenta(self, img, ret_angle=False, ret_y=False, ret_midx=False):
        uh_mag = 175; us_mag = 255; uv_mag = 255
        lh_mag = 150; ls_mag = 90; lv_mag = 110
        lower_hsv_mag = np.array([lh_mag, ls_mag, lv_mag])
        upper_hsv_mag = np.array([uh_mag, us_mag, uv_mag])

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        magenta_mask = cv2.inRange(hsv_img, lower_hsv_mag, upper_hsv_mag)

        contours, _ = cv2.findContours(magenta_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours.__len__() == 0:
            if not ret_angle and not ret_y and not ret_midx:
                return False
            elif ret_angle:
                return 0
            elif ret_y:
                return self.img_height - 1
            elif ret_midx:
                return 0
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        if self.state == 'truck':
            if y >= self.img_height - self.road_buffer:
                return True
            else: 
                return False
        
        elif self.state == 'desert':
            rect = cv2.minAreaRect(largest_contour)
            if ret_angle:
                return rect[2]
            elif ret_y:
                if cv2.contourArea(largest_contour) < self.desert_past_magenta_line_area:
                    return self.img_height -1
                else:
                    return rect[0][1]
            elif cv2.contourArea(largest_contour) > self.desert_min_magenta_area:
                return True
            else:
                return False
        
        elif self.state == 'yoda':
            if ret_midx:
                return x + w // 2
            elif ret_y:
                return y + h if cv2.contourArea(largest_contour) > self.yoda_mag_min_area_for_y else 0
            else: 
                return True if cv2.contourArea(largest_contour) > self.yoda_find_mag_min_area else False
            
    def thresh_desert(self, img):
        uh = 37; us = 98; uv = 255
        lh = 13; ls = 35; lv = 179
        lower_hsv = np.array([lh, ls, lv])
        upper_hsv = np.array([uh, us, uv])

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda contour: cv2.arcLength(contour, True), reverse=True)
        contours = [cnt for cnt in contours if cv2.arcLength(cnt, True) > self.desert_min_arc_length 
                    and cv2.boundingRect(cnt)[3] > self.desert_line_cnt_min_height ]
        if len(contours) == 0:
                    return np.zeros_like(img)
        epsilon = 0.01 * cv2.arcLength(contours[0], True)
        approx_cnts = [cv2.approxPolyDP(cnt, epsilon, True) for cnt in contours]

        blank_img = np.zeros_like(img)

        return cv2.fillPoly(blank_img, approx_cnts, (255, 255, 255))

    # returns true if cactus contour area within range
    def check_cactus(self, img):
        uh_cactus = 66; us_cactus = 255; uv_cactus = 255
        lh_cactus = 56; ls_cactus = 86; lv_cactus = 63
        lower_hsv_cactus = np.array([lh_cactus, ls_cactus, lv_cactus])
        upper_hsv_cactus = np.array([uh_cactus, us_cactus, uv_cactus])

        uh_yoda = 68; us_yoda = 255; uv_yoda = 255
        lh_yoda = 57; ls_yoda = 96; lv_yoda = 89
        lower_hsv_yoda = np.array([lh_yoda, ls_yoda, lv_yoda])
        upper_hsv_yoda = np.array([uh_yoda, us_yoda, uv_yoda])

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cactus_mask = cv2.inRange(hsv_img, lower_hsv_cactus, upper_hsv_cactus)
        yoda_mask = cv2.inRange(hsv_img, lower_hsv_yoda, upper_hsv_yoda)
        yoda_mask = cv2.bitwise_not(yoda_mask)

        mask = cv2.bitwise_and(cactus_mask, yoda_mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours.__len__() == 0:
            return False
        largest_contour = max(contours, key=cv2.contourArea)
        if self.cactus_min_area < cv2.contourArea(largest_contour) < self.cactus_max_area:
            return True
        else:
            return False
    
    # returns the centre point of the bounding rectangle of the tunnel, img width if no tunnel found by default
    # can also return the contour area and the mask image
    def find_tunnel(self, img, ret_area=False, ret_mask=False):
        uh = 9; us = 255; uv = 255
        lh = 0; ls = 106; lv = 66
        lower_hsv= np.array([lh, ls, lv])
        upper_hsv= np.array([uh, us, uv])

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)

        if ret_mask:
            return mask

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.tunnel_min_area]
        if len(contours) == 0:
            return -1 if not ret_area else 0
        combined_contour = np.concatenate(contours)
        x, y, w, h = cv2.boundingRect(combined_contour)
        return x + w // 2 if not ret_area else cv2.contourArea(combined_contour)

    # placeholder for start function
    def start(self):
        # start the timer
        print('starting timer, entering road pid state')
        # self.state = 'road'
        self.state = 'desert'
    
    # main loop for the driver
    def run(self):
        while not rospy.is_shutdown():
            if self.img is None:
                continue
            
            # --------------- initialization state ---------------
            elif self.state == 'init':
                self.start()

            # -------------------- road state --------------------
            elif self.state == 'road':
                if self.reached_crosswalk == False and self.check_red(self.img):
                    print('red detected, going to ped state')
                    self.state = 'ped'
                # elif self.reached_truck and self.check_magenta(self.img):
                #     print('magenta detected, going to desert state')
                #     self.state = 'desert'
                else:
                    error = self.kp * self.get_error(self.img)
                    # print(error)
                    self.drive_robot(self.lin_speed, self.rot_speed * error)

            # ----------------- pedestrian state -----------------
            elif self.state == 'ped':
                # angle to be straight on with crosswalk
                while self.red_line_min_angle < self.check_red(self.img, ret_angle=True) < self.red_line_max_angle:
                    angle = self.check_red(self.img, ret_angle=True)
                    if angle < 45:
                        self.drive_robot(self.red_line_approach_lin_vel, -1 * angle * self.red_line_approach_rot_vel)
                    else:
                        self.drive_robot(self.red_line_approach_lin_vel, (90 - angle) * self.red_line_approach_rot_vel)
                
                # get close to crosswalk
                while self.check_red(self.img, ret_y=True) < self.red_line_stop_y:
                    self.drive_robot(self.lin_speed, 0)

                self.drive_robot(0, 0)

                if self.check_pedestrian(self.img):
                    # print('pedestrian detected, waiting...')
                    self.ped_safe_count = 0
                else:
                    self.ped_safe_count += 1
                    if self.ped_safe_count > self.ped_safe_count_buffer:
                        # print('no pedestrian, going!')
                        self.drive_robot(self.ped_lin_speed, self.ped_ang_speed)
                        rospy.sleep(self.ped_sleep_time)
                        print('crossing crosswalk, going back to road pid state')
                        self.state = 'road'
                        self.reached_crosswalk = True
            
            # ------------------- truck state --------------------
            elif self.state == 'truck':
                if not self.reached_truck:
                    self.drive_robot(0, 0)
                    truck_area, truck_mid = self.check_truck(self.img, at_intersection=True)
                    if self.cycle_count < self.truck_init_cycle + self.truck_cycle_buffer:
                        # print('too early to tell')
                        pass
                    elif truck_mid < self.img_width // 2 and truck_area > self.truck_left_area:
                        print('truck close but on left, going right')
                        self.truck_turn_dir = 'right'
                        self.reached_truck = True
                    elif truck_area > self.truck_wait_area:
                        print('truck is close, waiting...')
                        self.drive_robot(0, 0)
                        self.truck_turn_dir = 'wait'
                    else:
                        print('going left')
                        self.truck_turn_dir = 'left'
                        self.reached_truck = True
                elif self.truck_turn_dir == 'right':
                    error = self.truck_right_kp * self.get_error(self.img)
                    self.drive_robot(self.truck_right_lin_speed, self.rot_speed * error)
                else:
                    error = self.kp * self.get_error(self.img)
                    self.drive_robot(self.lin_speed, self.rot_speed * error)

                if self.check_magenta(self.img):
                    print('magenta detected, going to desert state')
                    self.state = 'desert'
                    self.drive_robot(self.lin_speed, 0)
                    rospy.sleep(self.truck_to_desert_sleep)

            # ------------------ desert state --------------------
            elif self.state == 'desert':
                if self.check_magenta(self.img):
                    self.drive_robot(0, 0)
                    print('detected magenta')
                    while self.magneta_min_angle < self.check_magenta(self.img, ret_angle=True) < self.magneta_max_angle:
                        angle = self.check_magenta(self.img, ret_angle=True)
                        if angle < 45:
                            self.drive_robot(self.magenta_angle_lin_speed, -1 * angle * self.magenta_angle_rot_speed)
                        else:
                            self.drive_robot(self.magenta_angle_lin_speed, (90 - angle) * self.magenta_angle_rot_speed)

                    print('done angling, moving closer')
                    while self.check_magenta(self.img, ret_y=True) < self.img_height - 10:
                        self.drive_robot(self.magenta_angle_lin_speed, 0)
                   
                    self.drive_robot(0, 0)
                    print('going to yoda state')
                    self.state = 'yoda'
                else:
                    error = (self.kp) * self.get_error(self.img)
                    self.drive_robot(self.lin_speed, self.rot_speed * error)

            # -------------------- yoda state --------------------
            elif self.state == 'yoda':
                if not self.reached_yoda:
                    print('getting close to cactus')
                    while not self.check_cactus(self.img):
                        self.drive_robot(self.cactus_lin_speed, 0)
                    print('turning to see tunnel')
                    while self.find_tunnel(self.img) < self.tunnel_mid_x:
                        self.drive_robot(0, self.tunnel_turn_speed)
                    print('all good, ready to go over the hill')
                    self.drive_robot(0, 0)
                    self.reached_yoda = True
                else:
                    if not self.over_hill:
                        tunnel_mid = self.find_tunnel(self.img)
                        if tunnel_mid == -1:
                            while not self.check_magenta(self.img):
                                self.drive_robot(self.hill_lin_speed, self.hill_rot_speed)
                            print('over the hill now, checking for magenta')
                            self.over_hill = True
                        else:
                            error = self.kp * (self.tunnel_mid_x - tunnel_mid) / self.tunnel_mid_x
                            self.drive_robot(self.lin_speed, self.rot_speed * error)
                    else:
                        while self.check_magenta(self.img, ret_y=True) < self.yoda_mag_y_exit:
                            mag_x = self.check_magenta(self.img, ret_midx=True)
                            error = self.kp * (self.yoda_mag_x_mid - mag_x) / self.yoda_mag_x_mid
                            self.drive_robot(self.lin_speed, self.rot_speed * error)
                        print('close to magenta, angling to be straight')
                        # TODO: TEST BELOW THIS
                        while self.magneta_min_angle < self.check_magenta(self.img, ret_angle=True) < self.magneta_max_angle:
                            angle = self.check_magenta(self.img, ret_angle=True)
                            if angle < 45:
                                self.drive_robot(self.magenta_angle_lin_speed, -1 * angle * self.magenta_angle_rot_speed)
                            else:
                                self.drive_robot(self.magenta_angle_lin_speed, (90 - angle) * self.magenta_angle_rot_speed)
                        print('done angling, moving closer')
                        while self.check_magenta(self.img, ret_y=True) < self.img_height - 10:
                            self.drive_robot(self.magenta_angle_lin_speed, 0)
                        print('going to tunnel state')
                        self.state = 'tunnel'

            # ------------------ tunnel state ------------------
            elif self.state == 'tunnel':
                # TODO: TEST THIS
                # self.drive_robot(0, 0)
                if self.find_tunnel(self.img, ret_area=True) < 1000:
                    # pid with tunnel
                    error = self.kp * self.get_error(self.img)
                    self.drive_robot(self.lin_speed, self.rot_speed * error)
                else:
                    print('tunnel contour too small, going to mountain state')
                    self.state = 'mountain'

            # ----------------- mountain state -----------------
            elif self.state == 'mountain':
                self.drive_robot(0, 0)
        
        # rospy.sleep(0.1)

if __name__ == '__main__':
    try:
        my_driver = Driver()
        rospy.sleep(1)
        my_driver.run()
    except rospy.ROSInterruptException:
        pass