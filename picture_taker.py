#! /usr/bin/env python3

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2

class Picture_Taker():
    def __init__(self):
        rospy.init_node('picture_taker')

        self.bridge = CvBridge()
        self.img = None
        self.counter = 0

        rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback)

    def callback(self, msg):
        self.img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        # self.img = cv2.cvtColor(init_img, cv2.COLOR_BGR2RGB)

    def run(self):
        while True:
            if self.img is not None:
                cv2.imshow('image', self.img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    cv2.imwrite(f'img_{self.counter}.jpg', self.img)
                    self.counter += 1
                    print(f'saved image {self.counter}')
                elif key == ord('q'):
                    break
        return

        
if __name__ == '__main__':
    try:
        my_bot = Picture_Taker()
        rospy.sleep(1)
        my_bot.run()
    except rospy.ROSInterruptException:
        pass

