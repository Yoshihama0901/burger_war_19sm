#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from time import sleep
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import numpy as np
from gazebo_msgs.msg import ModelStates
import datetime

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
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.callback_model_state, queue_size=10)
        self.enemy_x = -1
        self.enemy_y = -1
        self.me_x = -1
        self.me_y = -1
        dt = datetime.datetime.now()
        self.log_fname = "circle-" + dt.strftime("%Y%m%d%H%M%S") + ".log"

    def callback_model_state(self, data):
        self.enemy_x = data.pose[37].position.x
        self.enemy_y = data.pose[37].position.y
        self.me_x = data.pose[36].position.x
        self.me_y = data.pose[36].position.y
        
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
        height = self.img.shape[0]
        canny_param = 100
        canny = cv2.Canny(red, canny_param/2, canny_param)
        circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT,
                                   dp=1, minDist=height/10, param1=canny_param, param2=8,
                                   minRadius=height/96, maxRadius=height/12)
        circle_x = -1
        circle_y = -1
        circle_r = -1
        if circles is not None:
            for i in circles[0,:]:
                x = int(i[0])
                y = int(i[1])
                r = int(i[2])
                if (y < height * 5 / 8) and (r > circle_r):
                    circle_x = x
                    circle_y = y
                    circle_r = r
        with open(self.log_fname, mode='a') as f:
            f.write('%f,%f,%f,%f,%d,%d,%d\n' % (self.me_x, self.me_y, self.enemy_x, self.enemy_y, circle_x, circle_y, circle_r))
        hough = self.img.copy()
        if circles is not None:
            for i in circles[0,:]:
                color = (255, 255, 0)
                pen_width = 2
                if circle_x == int(i[0]) and circle_y == int(i[1]):
                    color = (0, 255, 0)
                    pen_width = 4
                cv2.circle(hough, (int(i[0]), int(i[1])), int(i[2]), color, pen_width)
        if self.preview:
            #cv2.imshow("red", red)
            #cv2.imshow("canny", canny)
            cv2.imshow(self.w_name, hough)
            cv2.waitKey(1)

if __name__ == '__main__':
    iw = ImageWindow(w_name="imageview")
    rospy.init_node('imageview', anonymous=True)
    
    while(True):
        sleep(1)
