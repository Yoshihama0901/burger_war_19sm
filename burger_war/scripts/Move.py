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

from std_msgs.msg import String
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist, Vector3, Quaternion

# 強化学習DQN (Deep Q Network)
from MyModule import DQN


# クォータニオンからオイラー角への変換
def quaternion_to_euler(quaternion):
    e = tf.transformations.euler_from_quaternion((quaternion.x, quaternion.y, quaternion.z, quaternion.w))
    return Vector3(x=e[0]*180/np.pi, y=e[1]*180/np.pi, z=e[2]*180/np.pi)


class RandomBot():
    def __init__(self, bot_name, color='r'):
        self.name     = bot_name                                        # bot name 
        self.vel_pub  = rospy.Publisher('cmd_vel', Twist, queue_size=1) # velocity publisher
        self.sta_pub  = rospy.Publisher("/gazebo/model_states", ModelStates) # 初期化用
        self.state    = np.reshape(np.zeros(28), [1, 28])               # 状態
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
        for i in range(18):
            player = json_dict['targets'][i]['player']
            if player == self.my_color : self.score[2+i] =  float(json_dict['targets'][i]['point'])
            if player == self.en_color : self.score[2+i] = -float(json_dict['targets'][i]['point'])
        if self.my_color == 'b':                           # 自分が青色だった場合、相手と自分を入れ替える
            for i in range(3) : self.score[2+i], self.score[5+i] = self.score[5+i], self.score[2+i]
    
    # 位置情報の更新(model_stateのコールバック関数)
    def callback_model_state(self, data):
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
        if self.timer > 180 :
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
        
        # 審判情報の更新
        rospy.Subscriber("war_state", String, self.callback_war_state)
        score = ''
        for a in self.score : score += str(int(a)) + ' '
        #print('Score : ', score)
        
        # 位置情報
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.callback_model_state)
        my_angle = quaternion_to_euler(Quaternion(self.pos[2], self.pos[3], self.pos[4],  self.pos[5] ))
        en_angle = quaternion_to_euler(Quaternion(self.pos[8], self.pos[9], self.pos[10], self.pos[11]))
        #print('  Pos   : my-pos(%4.2f, %4.2f) my-angle(%4.0f), en-pos(%4.2f, %4.2f) en-angle(%4.0f)' % (self.pos[0], self.pos[1], my_angle.z, self.pos[6], self.pos[7], en_angle.z))
        
        # 状態と報酬の更新
        next_state = np.concatenate([self.pos[:8], self.score])
        next_state = np.reshape(next_state, [1, 28])          # 現在の状態(自分と相手の位置、点数)
        reward     =  self.calc_reward()
        
        # 行動を決定する
        action, linear, angle = self.actor.get_action(self.state, 1, self.mainQN)
        twist = Twist()
        twist.linear.y  = 0; twist.linear.z = 0; twist.angular.x = 0; twist.angular.y = 0
        twist.linear.x  = linear # 0.2
        twist.angular.z = angle  #   1
        
        # メモリの更新する
        self.memory.add((self.state, action, reward, next_state))  # メモリの更新する
        self.state = next_state                                    # 状態更新
        
        # Qネットワークの重みを学習・更新する replay
        learn = 1         # 学習を行うかのフラグ
        batch_size = 32   # Q-networkを更新するバッチの大きさ
        gamma = 0.99      # 割引係数
        if (self.memory.len() > batch_size) and learn:
            self.mainQN.replay(self.memory, batch_size, gamma, self.targetQN)
        self.targetQN.model.set_weights(self.mainQN.model.get_weights())
        
        print('Time:%3.0f  Reward:%3.1f  Linar:%4.1f  Angle:%2.0f' % (self.timer, reward, twist.linear.x, twist.angular.z))
        self.reward = reward
        
        return twist


    # シュミレーション再開
    def restart(self, r):
        self.memory.reset()
        subprocess.call('bash ../catkin_ws/src/burger_war/judge/test_scripts/init_single_play.sh ../catkin_ws/src/burger_war/judge/marker_set/sim.csv localhost:5000 you enemy', shell=True)
        r.sleep()
        self.timer    = 0


    def strategy(self):
        
        rospy_Rate = 1
        r = rospy.Rate(rospy_Rate) # １秒間に送る送信回数 (change speed 1fps)
        
        # Qネットワークとメモリ、Actorの生成--------------------------------------------------------
        hidden_size   = 64              # Q-networkの隠れ層のニューロンの数
        learning_rate = 0.0001          # Q-networkの学習係数
        #learning_rate = 0.00001         # Q-networkの学習係数
        memory_size   = 10000           # バッファーメモリの大きさ
        self.mainQN   = DQN.QNetwork(hidden_size=hidden_size, learning_rate=learning_rate)   # メインのQネットワーク
        self.targetQN = DQN.QNetwork(hidden_size=hidden_size, learning_rate=learning_rate)   # 価値を計算するQネットワーク
        self.memory   = DQN.Memory(max_size=memory_size)
        self.actor    = DQN.Actor()
        self.mainQN.model.load_weights('weight.hdf5')     # 重みの読み込み
        
        self.targetQN.model.set_weights(self.mainQN.model.get_weights())
        while not rospy.is_shutdown():
            
            twist = self.calcTwist()    # 移動距離と角度を計算
            self.vel_pub.publish(twist) # ROSに反映
            
            # 試合終了した場合
            if abs(self.reward) == 1 or self.timer > 180:
            #if abs(self.reward) == 1 or self.timer > 18:
                if   self.reward == 0 : print('Draw')
                elif self.reward == 1 : print('Win!')
                else                  : print('Lose')
                with open('result.csv', 'a') as f:
                    writer = csv.writer(f, lineterminator='\n')
                    writer.writerow([self.score[0], self.score[1]])
                self.mainQN.model.save_weights('weight.hdf5')            # モデルの保存
                self.restart(r)                                          # 試合再開
            
            r.sleep()


if __name__ == '__main__':
    
    rospy.init_node('IntegAI_run')    # 初期化宣言 : このソフトウェアは"IntegAI_run"という名前
    bot = RandomBot('Team Integ AI')
    
    bot.strategy()

