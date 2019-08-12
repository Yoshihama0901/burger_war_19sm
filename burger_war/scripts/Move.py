#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import tf
import csv
import json
import rospy
import random
import subprocess
import numpy as np
import sys

from std_msgs.msg import String
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist, Vector3, Quaternion, PoseWithCovarianceStamped

# 強化学習DQN (Deep Q Network)
from MyModule import DQN

timeScale = 4 # １秒間で何回座標計算するか？

# クォータニオンからオイラー角への変換
def quaternion_to_euler(quaternion):
    e = tf.transformations.euler_from_quaternion((quaternion.x, quaternion.y, quaternion.z, quaternion.w))
    return Vector3(x=e[0]*180/np.pi, y=e[1]*180/np.pi, z=e[2]*180/np.pi)


# 座標回転行列を返す
def get_rotation_matrix(rad):
    rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    return rot


# 現在地を２次元ベクトル(n*n)にして返す
def get_pos_matrix(x, y, n=16):
    #my_pos  = np.array([self.pos[0], self.pos[1]])  # 現在地点
    pos     = np.array([x, y])                      # 現在地点
    rot     = get_rotation_matrix(45 * np.pi / 180) # 45度回転行列の定義
    rotated = ( np.dot(rot, pos) / 3.4 ) + 0.5      # 45度回転して最大幅1.7で正規化(0-1)
    pos_np  = np.zeros([n, n])
    pos_np[int(rotated[0]*n)][int(rotated[1]*n)] = 1
    return pos_np


# 自分が向いている向きを２次元ベクトル(n*n)にして返す
def get_ang_matrix(my_pos, angle, n=16):
    #angle = my_angle.z + 22.5
    while angle > 0 : angle -= 360
    while angle < 0 : angle += 360
    #print(angle)
    my_ang  = np.zeros([n, n])
    my_pos_x = np.where(my_pos==1)[0][0]
    my_pos_y = np.where(my_pos==1)[1][0]
    try:
        if   0 <= angle <  45 : my_ang[my_pos_x+1, my_pos_y+1] = 1
        if  45 <= angle <  90 : my_ang[my_pos_x+0, my_pos_y+1] = 1
        if  90 <= angle < 135 : my_ang[my_pos_x-1, my_pos_y+1] = 1
        if 135 <= angle < 180 : my_ang[my_pos_x-1, my_pos_y+0] = 1
        if 180 <= angle < 225 : my_ang[my_pos_x-1, my_pos_y-1] = 1
        if 225 <= angle < 270 : my_ang[my_pos_x+0, my_pos_y-1] = 1
        if 270 <= angle < 315 : my_ang[my_pos_x+1, my_pos_y-1] = 1
        if 315 <= angle < 360 : my_ang[my_pos_x+1, my_pos_y+0] = 1
    except:
        my_ang[my_pos_x, my_pos_y] = 1
    return my_ang


# 得点ベクトルを返す
def get_sco_matrix(score, point):
    #point = 1
    np_sco = np.zeros([16, 16])
    if score[8]  == point : np_sco[ 3,  8] = 1   #  8:Tomato_N
    if score[9]  == point : np_sco[ 4,  7] = 1   #  9:Tomato_S
    if score[10] == point : np_sco[ 7, 12] = 1   # 10:Omelette_N
    if score[11] == point : np_sco[ 8, 11] = 1   # 11:Omelette_S
    if score[12] == point : np_sco[ 7,  4] = 1   # 12:Pudding_N
    if score[13] == point : np_sco[ 8,  3] = 1   # 13:Pudding_S
    if score[14] == point : np_sco[11,  8] = 1   # 14:OctopusWiener_N
    if score[15] == point : np_sco[12,  7] = 1   # 15:OctopusWiener_S
    if score[16] == point : np_sco[ 7,  8] = 1   # 16:FriedShrimp_N
    if score[17] == point : np_sco[ 7,  7] = 1   # 17:FriedShrimp_E
    if score[18] == point : np_sco[ 8,  8] = 1   # 18:FriedShrimp_W
    if score[19] == point : np_sco[ 8,  7] = 1   # 19:FriedShrimp_S
    return np_sco



class RandomBot():
    def __init__(self, bot_name, color='r'):
        self.name     = bot_name                                        # bot name 
        self.vel_pub  = rospy.Publisher('cmd_vel', Twist, queue_size=1) # velocity publisher
        self.sta_pub  = rospy.Publisher("/gazebo/model_states", ModelStates, latch=True) # 初期化用
        #self.state    = np.reshape(np.zeros(28), [1, 28])               # 状態
        self.state    = np.zeros([1, 16, 16, 9])           # 状態
        self.timer    = 0                                               # 対戦時間
        self.reward   = 0.0                                             # 報酬
        self.my_color = color                                           # 自分の色情報
        self.en_color = 'b' if color=='r' else 'r'                      # 相手の色情報
        self.score    = np.zeros(20)                                    # スコア情報(以下詳細)
         #  0:自分のスコア, 1:相手のスコア
         #  2:相手後ろ, 3:相手Ｌ, 4:相手Ｒ, 5:自分後ろ, 6:自分Ｌ, 7:自分Ｒ
         #  8:Tomato_N, 9:Tomato_S, 10:Omelette_N, 11:Omelette_S, 12:Pudding_N, 13:Pudding_S
         # 14:OctopusWiener_N, 15:OctopusWiener_S, 16:FriedShrimp_N, 17:FriedShrimp_E, 18:FriedShrimp_W, 19:FriedShrimp_S
        self.pos      = np.zeros(12)                                    # 位置情報(以下詳細)
         #  0:自分位置_x,  1:自分位置_y,  2:自分角度_x,  3:自分角度_y,  4:自分角度_z,  5:自分角度_w
         #  6:相手位置_x,  7:相手位置_y,  8:相手角度_x,  9:相手角度_y, 10:相手角度_z, 11:相手角度_w

    # スコア情報の更新(war_stateのコールバック関数)
    def callback_war_state(self, data):
        json_dict = json.loads(data.data)                  # json辞書型に変換
        self.score[0] = json_dict['scores'][self.my_color] # 自分のスコア
        self.score[1] = json_dict['scores'][self.en_color] # 相手のスコア
        if json_dict['state'] == 'running':
            for i in range(18):
                #print('*********', len(json_dict['targets']))
                player = json_dict['targets'][i]['player']
                if player == self.my_color : self.score[2+i] =  float(json_dict['targets'][i]['point'])
                if player == self.en_color : self.score[2+i] = -float(json_dict['targets'][i]['point'])
            if self.my_color == 'b':                           # 自分が青色だった場合、相手と自分を入れ替える
                for i in range(3) : self.score[2+i], self.score[5+i] = self.score[5+i], self.score[2+i]

    # 位置情報の更新(amcl_poseのコールバック関数)
    def callback_amcl_pose(self, data):
        pos = data.pose.pose.position
        ori = data.pose.pose.orientation
        self.pos[0] = pos.x; self.pos[1] = pos.y; self.pos[2] = ori.x; self.pos[3] = ori.y; self.pos[4] = ori.z; self.pos[5] = ori.w
    
    # 位置情報の更新(model_stateのコールバック関数)
    def callback_model_state(self, data):
        #print('*********', len(data.pose))
        pos = data.pose[37].position;    self.pos[0] = pos.x; self.pos[1] = pos.y;
        ori = data.pose[37].orientation; self.pos[2] = ori.x; self.pos[3] = ori.y; self.pos[4]  = ori.z; self.pos[5]  = ori.w
        pos = data.pose[36].position;    self.pos[6] = pos.x; self.pos[7] = pos.y;
        ori = data.pose[36].orientation; self.pos[8] = ori.x; self.pos[9] = ori.y; self.pos[10] = ori.z; self.pos[11] = ori.w
        if self.my_color == 'b':                           # 自分が青色だった場合、相手と自分を入れ替える
            for i in range(6) : self.pos[i], self.pos[6+i] = self.pos[6+i], self.pos[i]

    # 報酬の計算
    def calc_reward(self):
        # 部分点
        reward = 0.0
        reward = ( self.score[0] - self.score[1] ) / 10.0
        if reward >  1: reward =  1
        if reward < -1: reward = -1
        
        # 試合終了
        if self.timer > 180 * timeScale:
            if self.score[0] > self.score[1] : reward =  1
            if self.score[0] < self.score[1] : reward = -1
        if not self.score[2] == 0 : reward =  1  # 一本勝ち
        if not self.score[5] == 0 : reward = -1  # 一本負け
        
        return reward


    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    # _/ 行動計算のメイン部
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def calcTwist(self):
        
        self.timer += 1
        
        # 位置情報
        #rospy.Subscriber("/gazebo/model_states", ModelStates, self.callback_model_state, queue_size=10)
        if self.my_color == 'r' : rospy.Subscriber("/red_bot/amcl_pose",  PoseWithCovarianceStamped, self.callback_amcl_pose)
        if self.my_color == 'b' : rospy.Subscriber("/blue_bot/amcl_pose", PoseWithCovarianceStamped, self.callback_amcl_pose)
        my_angle = quaternion_to_euler(Quaternion(self.pos[2], self.pos[3], self.pos[4], self.pos[5]))
        my_pos = get_pos_matrix(self.pos[0], self.pos[1])  # 自分の位置
        en_pos = get_pos_matrix(self.pos[6], self.pos[7])  # 相手の位置
        my_ang = get_ang_matrix(my_pos, my_angle.z + 22.5) # 自分の向き
        #print(my_pos)
        
        # 審判情報の更新(点数)
        rospy.Subscriber("war_state", String, self.callback_war_state, queue_size=10)
        my_sco      = get_sco_matrix(self.score,  1)                           # 自分の点数
        en_sco      = get_sco_matrix(self.score, -1)                           # 相手の点数
        mySide1_sco = my_pos if not self.score[6] == 0 else np.zeros([16, 16]) # 自分側面１の点数
        mySide2_sco = my_pos if not self.score[7] == 0 else np.zeros([16, 16]) # 自分側面２の点数
        enSide1_sco = en_pos if not self.score[3] == 0 else np.zeros([16, 16]) # 相手側面１の点数
        enSide2_sco = en_pos if not self.score[4] == 0 else np.zeros([16, 16]) # 相手側面２の点数
        
        # 状態と報酬の更新( 16 × 16 × 9ch )
        next_state = np.concatenate([np.expand_dims(my_pos,      axis=2),
                                     np.expand_dims(en_pos,      axis=2),
                                     np.expand_dims(my_ang,      axis=2),
                                     np.expand_dims(my_sco,      axis=2),
                                     np.expand_dims(en_sco,      axis=2),
                                     np.expand_dims(mySide1_sco, axis=2),
                                     np.expand_dims(mySide2_sco, axis=2),
                                     np.expand_dims(enSide1_sco, axis=2),
                                     np.expand_dims(enSide2_sco, axis=2)], axis=2)
        next_state = np.reshape(next_state, [1, 16, 16, 9])       # 現在の状態(自分と相手の位置、点数)
        reward     =  self.calc_reward()                          # 報酬
        
        # 行動を決定する
        #action, linear, angle = self.actor.get_action(self.state, 1, self.mainQN)
        action, linear, angle = self.actor.get_action(self.state, self.timer/timeScale, self.mainQN)
        twist = Twist()
        twist.linear.y  = 0; twist.linear.z = 0; twist.angular.x = 0; twist.angular.y = 0
        twist.linear.x  = linear # 0.2
        twist.angular.z = angle  #   1
        
        if self.timer < 7:
             twist.linear.x  = 0.2
             twist.angular.z =   0
        
        # メモリの更新する
        self.memory.add((self.state, action, reward, next_state))  # メモリの更新する
        self.state = next_state                                    # 状態更新
        
        # Qネットワークの重みを学習・更新する replay
        learn = 1
        if self.my_color == 'b' : learn = 0
        #batch_size = 32   # Q-networkを更新するバッチの大きさ
        batch_size = 10   # Q-networkを更新するバッチの大きさ
        gamma = 0.99      # 割引係数
        if (self.memory.len() > batch_size) and learn:
            self.mainQN.replay(self.memory, batch_size, gamma, self.targetQN)
        self.targetQN.model.set_weights(self.mainQN.model.get_weights())
        
        #print('Time:%3.0f  Reward:%3.1f  Linar:%4.1f  Angle:%2.0f' % (self.timer, reward, twist.linear.x, twist.angular.z))
        sys.stdout.flush()
        self.reward = reward
        
        return twist


    # シュミレーション再開
    def restart(self, r):
        self.memory.reset()
        self.score  = np.zeros(20)
        self.timer  = 0
        self.reward = 0
        subprocess.call('bash ../catkin_ws/src/burger_war/burger_war/scripts/reset_state.sh', shell=True)
        r.sleep()


    def strategy(self):
        
        rospy_Rate = timeScale
        r = rospy.Rate(rospy_Rate) # １秒間に送る送信回数 (change speed 1fps)
        
        # Qネットワークとメモリ、Actorの生成--------------------------------------------------------
        learning_rate = 0.0001          # Q-networkの学習係数
        #learning_rate = 0.00001         # Q-networkの学習係数
        memory_size   = 10000           # バッファーメモリの大きさ
        self.mainQN   = DQN.QNetwork(learning_rate=learning_rate)   # メインのQネットワーク
        self.targetQN = DQN.QNetwork(learning_rate=learning_rate)   # 価値を計算するQネットワーク
        self.memory   = DQN.Memory(max_size=memory_size)
        self.actor    = DQN.Actor()
        self.mainQN.model.load_weights('../catkin_ws/src/burger_war/burger_war/scripts/weight.hdf5')     # 重みの読み込み
        
        self.targetQN.model.set_weights(self.mainQN.model.get_weights())
        while not rospy.is_shutdown():
            
            twist = self.calcTwist()    # 移動距離と角度を計算
            self.vel_pub.publish(twist) # ROSに反映
            
            # 試合終了した場合
            if self.my_color == 'r':
                if abs(self.reward) == 1 or self.timer > 180 * timeScale:
                    self.vel_pub.publish(Twist()) # 動きを止める
                    if   self.reward == 0 : print('Draw')
                    elif self.reward == 1 : print('Win!')
                    else                  : print('Lose')
                    with open('result.csv', 'a') as f:
                        writer = csv.writer(f, lineterminator='\n')
                        writer.writerow([self.score[0], self.score[1]])
                    self.mainQN.model.save_weights('../catkin_ws/src/burger_war/burger_war/scripts/weight.hdf5')            # モデルの保存
                    self.restart(r)                                          # 試合再開
            else:
                if self.timer % (180 * timeScale) == 0 : self.mainQN.model.load_weights('../catkin_ws/src/burger_war/burger_war/scripts/weight.hdf5')                # 重みの読み込み
            
            r.sleep()


if __name__ == '__main__':
    
    color = 'r'
    
    rospy.init_node('IntegAI_run')    # 初期化宣言 : このソフトウェアは"IntegAI_run"という名前
    bot = RandomBot('Team Integ AI', color=color)
    
    bot.strategy()

