#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import rospy
import random

from std_msgs.msg import String
from geometry_msgs.msg import Twist

#from std_msgs.msg import String
#from sensor_msgs.msg import Image
#from cv_bridge import CvBridge, CvBridgeError
#import cv2


class RandomBot():
    def __init__(self, bot_name):
        self.name = bot_name                                           # bot name 
        self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1) # velocity publisher

    def calcTwist(self):
        '''
        value = random.randint(1,1000)
        if value < 250:
            x = 0.2
            th = 0
        elif value < 500:
            x = -0.2
            th = 0
        elif value < 750:
            x = 0
            th = 1
        elif value < 1000:
            x = 0
            th = -1
        else:
            x = 0
            th = 0
        '''
        
        
        twist = Twist()
        twist.linear.y  = 0; twist.linear.z = 0; twist.angular.x = 0; twist.angular.y = 0
        twist.linear.x  = -0.2
        twist.angular.z = 0
        return twist


    def strategy(self):
        
        rospy_Rate = 10
        r = rospy.Rate(rospy_Rate) # ‚P•bŠÔ‚É‘—‚é‘—M‰ñ” (change speed 1fps)

        target_speed  = 0
        target_turn   = 0
        control_speed = 0
        control_turn  = 0

        while not rospy.is_shutdown():
            
            twist = self.calcTwist()    # ˆÚ“®‹——£‚ÆŠp“x‚ðŒvŽZ
            self.vel_pub.publish(twist) # ROS‚É”½‰f
            print(twist.linear.x, twist.angular.z)
            listener()
            
            r.sleep()


def callback(data):
    print(data.data)
    #rospy.loginfo(rospy.get_caller_id()+"I heard %s",data.data)


def listener():
    rospy.Subscriber("war_state", String, callback)


if __name__ == '__main__':
    
    rospy.init_node('IntegAI_run')    # ‰Šú‰»éŒ¾ : ‚±‚Ìƒ\ƒtƒgƒEƒFƒA‚Í"IntegAI_run"‚Æ‚¢‚¤–¼‘O
    bot = RandomBot('Team Integ AI')
    
    
    topic_list = rospy.get_published_topics('/')
    for a in topic_list : print(a)
    
    bot.strategy()

