#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from time import sleep
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

class ImageWindow:
    def __init__(self, w_name=None, window_size=(640,480)):
        self.w_name = w_name
        # cv2.namedWindow(self.w_name, cv2.WINDOW_NORMAL)
        # cv2.moveWindow(self.w_name, 100, 100)
        self.image_pub = rospy.Publisher('/red_bot/image_raw', Image, queue_size=10)
        self.img = None
        self.preview = True
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/red_bot/image_raw', Image, self.imageCallback, queue_size=10)

        
    def imageCallback(self, data):
        print('callback')
        try:
            self.img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        print('callback2')
        if self.preview:
            cv2.imshow(self.w_name, self.img)
            cv2.waitKey(1)
        
if __name__ == '__main__':
    iw = ImageWindow(w_name="imageview")
    rospy.init_node('imageview', anonymous=True)
    
    while(True):
        sleep(1)
