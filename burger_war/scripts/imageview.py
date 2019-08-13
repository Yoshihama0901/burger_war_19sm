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
import tf.transformations
from geometry_msgs.msg import Vector3, Quaternion

# クォータニオンからオイラー角への変換
def quaternion_to_euler(quaternion):
    e = tf.transformations.euler_from_quaternion((quaternion.x, quaternion.y, quaternion.z, quaternion.w))
    return Vector3(x=e[0]*180/np.pi, y=e[1]*180/np.pi, z=e[2]*180/np.pi)

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
        self.enemy_qx = -1
        self.enemy_qy = -1
        self.enemy_qz = -1
        self.enemy_qw = -1
        self.my_x = -1
        self.my_y = -1
        self.my_qx = -1
        self.my_qy = -1
        self.my_qz = -1
        self.my_qw = -1
        self.log_fname = None
        #self.log_fname = "circle-" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".csv"
        if self.log_fname is not None:
            with open(self.log_fname, mode='a') as f:
                f.write('my_x,my_y,my_qx,my_qy,my_qz,my_qw,my_ax,my_ay,my_az,enemy_x,enemy_y,enemy_qx,enemy_qy,enemy_qz,enemy_qw,enemy_ax,enemy_ay,enemy_az,circle_x,circle_y,circle_r\n')

    def callback_model_state(self, data):
        self.enemy_x = data.pose[36].position.x
        self.enemy_y = data.pose[36].position.y
        self.enemy_qx = data.pose[36].orientation.x
        self.enemy_qy = data.pose[36].orientation.y
        self.enemy_qz = data.pose[36].orientation.z
        self.enemy_qw = data.pose[36].orientation.w
        self.my_x = data.pose[37].position.x
        self.my_y = data.pose[37].position.y
        self.my_qx = data.pose[37].orientation.x
        self.my_qy = data.pose[37].orientation.y
        self.my_qz = data.pose[37].orientation.z
        self.my_qw = data.pose[37].orientation.w
        
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
        if self.log_fname is not None:
            with open(self.log_fname, mode='a') as f:
                my_angle = quaternion_to_euler(Quaternion(self.my_qx, self.my_qy, self.my_qz, self.my_qw))
                enemy_angle = quaternion_to_euler(Quaternion(self.enemy_qx, self.enemy_qy, self.enemy_qz, self.enemy_qw))
                f.write('%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d,%d,%d\n'
                        % (self.my_x, self.my_y, self.my_qx, self.my_qy, self.my_qz, self.my_qw,
                           my_angle.x, my_angle.y, my_angle.z,
                           self.enemy_x, self.enemy_y, self.enemy_qx, self.enemy_qy, self.enemy_qz, self.enemy_qw,
                           enemy_angle.x, enemy_angle.y, enemy_angle.z,
                           circle_x, circle_y, circle_r))
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
