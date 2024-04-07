#! /usr/bin/env python3

import rospy
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

class Accel():
    def __init__(self):
        rospy.init_node('robot_accel')
        self.vel_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
        self.bridge = CvBridge()
        rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.image_callback)
        self.max_vel = 2.5
        self.img = None

    def image_callback(self, data):
        self.img = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        # print('new image')

    def accel(self, target_vel):
        rate = rospy.Rate(60)
        vel = Twist()
        vel.linear.x = 0
        while vel.linear.x < target_vel:
            vel.linear.x += 0.05
            if vel.linear.x > target_vel:
                vel.linear.x = target_vel
            self.vel_pub.publish(vel)
            rate.sleep()

    def decel(self, target_vel):
        rate = rospy.Rate(10)
        vel = Twist()
        vel.linear.x = target_vel
        while vel.linear.x > 0:
            vel.linear.x -= 0.05
            if vel.linear.x < 0:
                vel.linear.x = 0
            self.vel_pub.publish(vel)
            rate.sleep()

    def accel2(self, target_vel):
        rate = rospy.Rate(50)
        vel = Twist()
        vel.linear.x = 0
        count = 1
        while vel.linear.x < target_vel:
            vel.linear.x += 0.05 * count
            if vel.linear.x > target_vel:
                vel.linear.x = target_vel
            self.vel_pub.publish(vel)
            count = count ** 2
            rate.sleep()

    def stop(self):
        vel = Twist()
        vel.linear.x = 0
        self.vel_pub.publish(vel)

    def run(self):
        self.accel2(self.max_vel)
        print('hit max vel')
        image = self.img
        cv2.imshow('image 1', image)
        cv2.waitKey(0)
        self.stop()
        print('stopped')
        self.accel(self.max_vel)
        print('hit max vel')
        cv2.imshow('image 2', self.img)
        cv2.waitKey(0)
        self.stop()

if __name__ == '__main__':
    try:
        accel = Accel()
        accel.run()
    except rospy.ROSInterruptException:
        pass