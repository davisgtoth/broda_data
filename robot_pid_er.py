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
        self.img_height = 0
        self.img_width = 0
        self.cycle_count = 0

        self.road_buffer = 200 # pixels above bottom of image to find road centre
        self.red_line_cutoff = 700 # pixels from top of image detect red line
        self.red_line_min_area = 1000 # minimum contour area for red line

        self.move = Twist()
        self.kp = 10 # proportional gain for PID controller
        self.lin_speed = 0.4 # defualt PID linear speed of robot
        self.rot_speed = 1.0 # base PID angular speed of robot
        self.speed_buffer = 1.5 # buffer for gradual speed increase/decrease

        self.bg_sub = cv2.createBackgroundSubtractorMOG2()
        self.reached_crosswalk = False
        self.ped_buffer = 60 # lateral pixel buffer for pedestrian detection

        self.ped_lin_speed = 2.0 # linear speed of robot when crossing crosswalk
        self.ped_ang_speed = 0 # angular speed of robot when crossing crosswalk
        self.ped_sleep_time = 0.6 # time to sleep when crossing crosswalk

        self.state = 'init' # init, road, ped, truck, desert, yoda

    # callback function for camera subscriber
    def callback(self, msg):
        self.img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.img_height, self.img_width = self.img.shape[:2]
        self.cycle_count += 1

    def find_road_centre(self, img, y, ret_sides=False):
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
        for i in range(self.img_width):
            if img[self.img_height - y, i] == 255 and left_index == -1:
                left_index = i
            elif img[self.img_height - y, i] == 255 and left_index != -1:
                right_index = i

        if ret_sides:
            return left_index, right_index

        road_centre = -1
        if left_index != -1 and right_index != -1:
            if right_index - left_index > 150:
                road_centre = (left_index + right_index) // 2
            elif left_index < self.img_width // 2:
                road_centre = (left_index + self.img_width) // 2
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
    def get_error(self, img, road=True, desert=False):
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
        if road:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = cv2.inRange(gray_img, 250, 255)
        elif desert:
            # TODO: threshold desert image
            mask = cv2.inRange(img, (0, 0, 0), (255, 255, 255)) # to be changed

        road_centre = self.find_road_centre(mask, self.road_buffer)
        if road_centre != -1:
            error = ((self.img_width // 2) - road_centre) / (self.img_width // 2)
        elif self.reached_crosswalk:
            error = 0
            print('no road detected, going to truck state')
            self.state = 'truck'
        else:
            error = 0
        return error
    
    def check_red(self, img):
        """
        Checks if red is found in the image with an area greater than red_line_min_area.

        Parameters:
        img (numpy.ndarray): The input image.

        Returns:
        bool: Returns True if red is found in the image with an area greater than red_line_min_area, otherwise False.
        """
        cropped_img = img[self.red_line_cutoff:self.img_height]
            
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
        if cv2.contourArea(largest_contour) < self.red_line_min_area:
            return False
        else:
            return True

    # return true if the pedestrian is on the cross walk or within the 
    def check_pedestrian(self, img):
        """
        Checks if a pedestrian is on the crosswalk or within a buffer distance to the road.

        Parameters:
        img (numpy.ndarray): The input image.

        Returns:
        bool: Returns True if a pedestrian is detected on the crosswalk or within the buffer, otherwise False.
        """
        fg_mask = self.bg_sub.apply(img)

        # cv2.imshow('camera feed', fg_mask)
        # cv2.waitKey(1)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours.__len__() == 0:
            return True
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        road_left, road_right = self.find_road_centre(img, self.img_height-(y+h-1), ret_sides=True)

        if road_left - self.ped_buffer < (x + w//2) < road_right + self.ped_buffer:
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
            vel = 0.2
            temp_counter = self.cycle_count
            while vel < linear:
                if self.cycle_count > temp_counter:
                    self.move.linear.x = vel
                    self.move.angular.z = angular
                    self.vel_pub.publish(self.move)
                    vel += 0.1
                    temp_counter = self.cycle_count

        elif linear < self.move.linear.x - self.speed_buffer:
            vel = self.move.linear.x
            temp_counter = self.cycle_count
            while vel < linear:
                if self.cycle_count > temp_counter:
                    self.move.linear.x = vel
                    self.move.angular.z = angular
                    self.vel_pub.publish(self.move)
                    vel -= 0.1
                    temp_counter = self.cycle_count
        else:
            self.move.linear.x  = linear
            self.move.angular.z = angular
            self.vel_pub.publish(self.move)

    # placeholder for start function
    def start(self):
        # start the timer
        print('starting timer, entering road pid state')
        self.state = 'road'
    
    # main loop for the driver
    def run(self):
        while not rospy.is_shutdown():
            if self.img is None:
                continue
            
            # initialization state
            elif self.state == 'init':
                self.start()

            # road state
            elif self.state == 'road':
                if self.reached_crosswalk == False and self.check_red(self.img):
                    print('red detected, going to ped state')
                    self.reached_crosswalk = True
                    self.state = 'ped'
                else:
                    error = self.kp * self.get_error(self.img)
                    # print(error)
                    self.drive_robot(self.lin_speed, self.rot_speed * error)

            # pedestrian state
            elif self.state == 'ped':
                if self.check_pedestrian(self.img):
                    print('pedestrian detected, waiting...')
                else:
                    print('no pedestrian, going!')
                    self.drive_robot(self.ped_lin_speed, self.ped_ang_speed)
                    rospy.sleep(self.ped_sleep_time)
                    print('crossing crosswalk, going back to road pid state')
                    self.state = 'road'
            
            # truck state
            elif self.state == 'truck':
                self.drive_robot(0, 0)

            # desert state
            elif self.state == 'desert':
                pass

            # yoda state
            elif self.state == 'yoda':
                pass
        
        rospy.sleep(0.1)

if __name__ == '__main__':
    try:
        my_driver = Driver()
        rospy.sleep(1)
        my_driver.run()
    except rospy.ROSInterruptException:
        pass
                