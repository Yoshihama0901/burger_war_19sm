#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from time import sleep
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import numpy as np

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
        try:
            self.img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV_FULL)
        hsv_h = hsv[:, :, 0]
        hsv_s = hsv[:, :, 1]
        mask = np.zeros(hsv_h.shape, dtype=np.uint8)
        mask[((hsv_h < 16) | (hsv_h > 240)) & (hsv_s > 64)] = 255
        red = cv2.bitwise_and(self.img, self.img, mask=mask)
        canny_th0 = 50
        canny_th1 = 100
        canny = cv2.Canny(red, canny_th0, canny_th1)
        circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT,
                                   dp=1, minDist=50, param1=canny_th1, param2=30,
                                   minRadius=5, maxRadius=20)
        # print(circles)
        hough = self.img.copy()
        if circles is not None:
            for i in circles[0,:]:
                cv2.circle(hough, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), 4)
        if self.preview:
            cv2.imshow("red", red)
            cv2.imshow("canny", canny)
            cv2.imshow(self.w_name, hough)
            cv2.waitKey(1)

if __name__ == '__main__':
    iw = ImageWindow(w_name="imageview")
    rospy.init_node('imageview', anonymous=True)
    
    while(True):
        sleep(1)
