#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import rospy
import random
import numpy as np

from std_msgs.msg import String
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist

#from std_msgs.msg import String
#from sensor_msgs.msg import Image
#from cv_bridge import CvBridge, CvBridgeError
#import cv2


class RandomBot():
    def __init__(self, bot_name, color='r'):
        self.name = bot_name                                            # bot name 
        self.vel_pub  = rospy.Publisher('cmd_vel', Twist, queue_size=1) # velocity publisher
        self.my_color = color                                           # �����̐F���
        self.en_color = 'b' if color=='r' else 'r'                      # ����̐F���
        self.score    = np.zeros(20)                                    # �X�R�A���(�ȉ��ڍ�)
                                                                        #  0 : �����̃X�R�A
                                                                        #  1 : ����̃X�R�A
                                                                        #  2 : ������
                                                                        #  3 : ����k
                                                                        #  4 : ����q
                                                                        #  5 : �������
                                                                        #  6 : �����k
                                                                        #  7 : �����q
                                                                        #  8 : Tomato_N
                                                                        #  9 : Tomato_S
                                                                        # 10 : Omelette_N
                                                                        # 11 : Omelette_S
                                                                        # 12 : Pudding_N
                                                                        # 13 : Pudding_S
                                                                        # 14 : OctopusWiener_N
                                                                        # 15 : OctopusWiener_S
                                                                        # 16 : FriedShrimp_N
                                                                        # 17 : FriedShrimp_E
                                                                        # 18 : FriedShrimp_W
                                                                        # 19 : FriedShrimp_S

    # �X�R�A���̍X�V(war_state�̃R�[���o�b�N�֐�)
    def callback_war_state(self, data):
        json_dict = json.loads(data.data)                  # json�����^�ɕϊ�
        self.score[0] = json_dict['scores'][self.my_color] # �����̃X�R�A
        self.score[1] = json_dict['scores'][self.en_color] # ����̃X�R�A
        for i in range(18):
            player = json_dict['targets'][i]['player']
            if player == self.my_color : self.score[2+i] =  float(json_dict['targets'][i]['point'])
            if player == self.en_color : self.score[2+i] = -float(json_dict['targets'][i]['point'])
        if self.my_color == 'b':                           # �������F�������ꍇ�A����Ǝ��������ւ���
            for i in range(3) : self.score[2+i], self.score[5+i] = self.score[5+i], self.score[2+i]
    
    # �ʒu���̍X�V(model_state�̃R�[���o�b�N�֐�)
    def callback_model_state(self, data):
        print(data)
        #print('ModelStates')

    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    # _/ �s���v�Z�̃��C����
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def calcTwist(self):
        
        # �R�����̍X�V
        rospy.Subscriber("war_state", String, self.callback_war_state)
        print('Score : ', self.score)
        
        # �ʒu���
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.callback_model_state)
        
        # ROS�s�����v�Z
        twist = Twist()
        twist.linear.y  = 0; twist.linear.z = 0; twist.angular.x = 0; twist.angular.y = 0
        twist.linear.x  = -0.2
        twist.angular.z = 0
        
        return twist


    def strategy(self):
        
        rospy_Rate = 10
        r = rospy.Rate(rospy_Rate) # �P�b�Ԃɑ��鑗�M�� (change speed 1fps)

        target_speed  = 0
        target_turn   = 0
        control_speed = 0
        control_turn  = 0

        while not rospy.is_shutdown():
            
            twist = self.calcTwist()    # �ړ������Ɗp�x���v�Z
            self.vel_pub.publish(twist) # ROS�ɔ��f
            print(twist.linear.x, twist.angular.z)
            
            r.sleep()


if __name__ == '__main__':
    
    rospy.init_node('IntegAI_run')    # �������錾 : ���̃\�t�g�E�F�A��"IntegAI_run"�Ƃ������O
    bot = RandomBot('Team Integ AI')
    
    
    topic_list = rospy.get_published_topics('/')
    for a in topic_list : print(a)
    
    bot.strategy()

