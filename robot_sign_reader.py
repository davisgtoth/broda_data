#! /usr/bin/env python3

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

import sign_cropper
import robot_pid_er

import tensorflow as tf
from tensorflow import keras as ks
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

# sess1 = tf.compat.v1.Session()   
# graph1 = tf.compat.v1.get_default_graph()
# set_session(sess1)
# tf.saved_model.LoadOptions(experimental_io_device = "/job:localhost")


class SignReader():
    def __init__(self):
        #rospy.init_node('sign_reader')

        self.bridge = CvBridge()
        rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback)

        self.vel_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)

        self.img = None
        self.min_sign_area = 6000

        self.nn = load_model('broda_data/my_model04')
        
        self.num_pixels_above_bottom = 200
        self.kp = 5
        self.lin_speed = 0.2
        self.rot_speed = 1.0
        self.no_lines_error = 1000
        
        self.sign_img = None

        self.num_signs = 0

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
            numpy.ndarray or None: The cropped image of the sign scaled and true size if found, None otherwise.
        """
        height, width = img.shape[:2]

        # threshold camera image for blue
        lower_hsv = (5,20,0)
        upper_hsv = (150,255,255)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)

        # threshold camera image for white
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sign_mask1 = cv2.inRange(gray_img, 95, 105)
        sign_mask2 = cv2.inRange(gray_img, 195, 205)
        sign_mask3 = cv2.inRange(gray_img, 115, 125)
        sign_mask = cv2.bitwise_or(sign_mask1, sign_mask2)
        sign_mask = cv2.bitwise_or(sign_mask, sign_mask3)

        # combine masks
        blue_mask_not = cv2.bitwise_not(blue_mask)
        combined_mask = cv2.bitwise_and(blue_mask_not, sign_mask)

        # find largest contour in the combined mask image
        contours, hierarchy = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        epsilon = 0.03 * cv2.arcLength(largest_contour, True)
        approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

        corners = [point[0] for point in approx_polygon]
        midpoint = int(len(corners)/2)
        sorted_corner_points = sorted(corners, key=lambda point: point[0])
        left = sorted(sorted_corner_points[:midpoint], key=lambda point: point[1])
        right = sorted(sorted_corner_points[midpoint:], key=lambda point: point[1], reverse=True)

        upperLeft = max((pt for pt in left), key=lambda p: p[1])
        lowerLeft = min((pt for pt in left), key=lambda p: p[1])
        upperRight = max((pt for pt in right), key=lambda p: p[1])
        lowerRight = min((pt for pt in right), key=lambda p: p[1])

        # perspective transform and crop the image
        src_pts = np.array([lowerLeft, upperLeft, lowerRight, upperRight], dtype=np.float32)
        dst_pts = np.array([[0, 0], [0, height], [width, 0], [width, height]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        cropped_img = cv2.warpPerspective(img, M, (width, height))

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
            print(self.sign_img.size)
        else:
            if new_sign.size > self.sign_img.size: # compare size of new sign to stored sign
                self.sign_img = new_sign
                print('bigger sign found')
        return
    
    def num_to_alphanum(self, x):
        if x <= 25:
            return chr(x + 65)
        else:
            return chr(x + 22)

    # when enough time has elapsed from initial sign detection, get the letters from the best 
    # sign image and send them to the neural network
    def read_sign(self):
        self.num_signs += 1
        category, clue = sign_cropper.signToLetters(self.sign_img)
        preds = []
        for i in range(clue.shape[0]):
            img_aug = np.expand_dims(clue[i], axis=0)
            letter = tf.expand_dims(img_aug, axis=-1)
            yp = self.nn.predict(letter)[0]
            predict_ind = np.argmax(yp)
            pred = self.num_to_alphanum(int(predict_ind))
            print(pred)
            preds.append(pred)
            
        prediction = ''.join(preds)
        print(str(prediction))
        return prediction
    
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
                cropped_img, cropped_img_true = self.check_if_sign(self.img) # returns None if no sign detected
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
                #cv2.imshow('sign', self.sign_img)
                #cv2.waitKey(1)
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
