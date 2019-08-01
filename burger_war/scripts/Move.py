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
        self.my_color = color                                           # 自分の色情報
        self.en_color = 'b' if color=='r' else 'r'                      # 相手の色情報
        self.score    = np.zeros(20)                                    # スコア情報(以下詳細)
                                                                        #  0 : 自分のスコア
                                                                        #  1 : 相手のスコア
                                                                        #  2 : 相手後ろ
                                                                        #  3 : 相手Ｌ
                                                                        #  4 : 相手Ｒ
                                                                        #  5 : 自分後ろ
                                                                        #  6 : 自分Ｌ
                                                                        #  7 : 自分Ｒ
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

    # スコア情報の更新(war_stateのコールバック関数)
    def callback_war_state(self, data):
        json_dict = json.loads(data.data)                  # json辞書型に変換
        self.score[0] = json_dict['scores'][self.my_color] # 自分のスコア
        self.score[1] = json_dict['scores'][self.en_color] # 相手のスコア
        for i in range(18):
            player = json_dict['targets'][i]['player']
            if player == self.my_color : self.score[2+i] =  float(json_dict['targets'][i]['point'])
            if player == self.en_color : self.score[2+i] = -float(json_dict['targets'][i]['point'])
        if self.my_color == 'b':                           # 自分が青色だった場合、相手と自分を入れ替える
            for i in range(3) : self.score[2+i], self.score[5+i] = self.score[5+i], self.score[2+i]
    
    # 位置情報の更新(model_stateのコールバック関数)
    def callback_model_state(self, data):
        print(data)
        #print('ModelStates')

    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    # _/ 行動計算のメイン部
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def calcTwist(self):
        
        # 審判情報の更新
        rospy.Subscriber("war_state", String, self.callback_war_state)
        print('Score : ', self.score)
        
        # 位置情報
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.callback_model_state)
        
        # ROS行動情報計算
        twist = Twist()
        twist.linear.y  = 0; twist.linear.z = 0; twist.angular.x = 0; twist.angular.y = 0
        twist.linear.x  = -0.2
        twist.angular.z = 0
        
        return twist


    def strategy(self):
        
        rospy_Rate = 10
        r = rospy.Rate(rospy_Rate) # １秒間に送る送信回数 (change speed 1fps)

        target_speed  = 0
        target_turn   = 0
        control_speed = 0
        control_turn  = 0

        while not rospy.is_shutdown():
            
            twist = self.calcTwist()    # 移動距離と角度を計算
            self.vel_pub.publish(twist) # ROSに反映
            print(twist.linear.x, twist.angular.z)
            
            r.sleep()


if __name__ == '__main__':
    
    rospy.init_node('IntegAI_run')    # 初期化宣言 : このソフトウェアは"IntegAI_run"という名前
    bot = RandomBot('Team Integ AI')
    
    
    topic_list = rospy.get_published_topics('/')
    for a in topic_list : print(a)
    
    bot.strategy()

