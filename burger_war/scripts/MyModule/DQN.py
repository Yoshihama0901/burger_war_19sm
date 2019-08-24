# -*- coding: utf-8 -*-

# _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
# _/  強化学習DQN (Deep Q Network)
# _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/

import numpy as np

from collections import deque
import tensorflow as tf
from keras import backend as K
from keras.utils import plot_model
from keras.models import Model
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam

# 次の行動を決める
def action_select(action):
    velocity = 0.5
    if action == 0 : linear = -velocity; angle = -1.0
    if action == 1 : linear = -velocity; angle =  0.0
    if action == 2 : linear = -velocity; angle =  1.0
    if action == 3 : linear =       0.0; angle = -1.0
    if action == 4 : linear =       0.0; angle =  0.0
    if action == 5 : linear =       0.0; angle =  1.0
    if action == 6 : linear =  velocity; angle = -1.0
    if action == 7 : linear =  velocity; angle =  0.0
    if action == 8 : linear =  velocity; angle =  1.0
    return linear, angle

# [1]損失関数の定義
# 損失関数にhuber関数を使用します 参考https://github.com/jaara/AI-blog/blob/master/CartPole-DQN.py
def huberloss(y_true, y_pred):
    err = y_true - y_pred
    cond = K.abs(err) < 1.0
    L2 = 0.5 * K.square(err)
    L1 = (K.abs(err) - 0.5)
    loss = tf.where(cond, L2, L1)  # Keras does not cover where function in tensorflow :-(
    return K.mean(loss)




# _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
# U-Net
#   https://qiita.com/koshian2/items/603106c228ac6b7d8356
# _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
# U-Net
def create_block(input, chs):
    x = input
    for i in range(2):
        # オリジナルはpaddingなしだがサイズの調整が面倒なのでPaddingを入れる
        x = Conv2D(chs, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    return x


def create_unet(size=16, use_skip_connections=True, grayscale_inputs=True):
    
    #if grayscale_inputs: input = Input((96,96,1))
    #else:                input = Input((96,96,3))
    input = Input((16, 16, 7))
    
    # Encoder
    block1 = create_block(input, 64)
    x = MaxPool2D(2)(block1)
    block2 = create_block(x, 128)
    x = MaxPool2D(2)(block2)
    block3 = create_block(x, 256)
    x = MaxPool2D(2)(block3)
    
    x = create_block(x, 512)
    x = Conv2DTranspose(256, kernel_size=2, strides=2)(x)
    if use_skip_connections: x = Concatenate()([block3, x])
    x = create_block(x, 256)
    x = Conv2DTranspose(128, kernel_size=2, strides=2)(x)
    if use_skip_connections: x = Concatenate()([block2, x])
    x = create_block(x, 128)
    x = Conv2DTranspose(64, kernel_size=2, strides=2)(x)
    if use_skip_connections: x = Concatenate()([block1, x])
    x = create_block(x, 64)
    
    # output
    x = Conv2D(1, 1)(x)
    #x = Activation("sigmoid")(x)
    #x = Activation("linear")(x)
    x = Activation("tanh")(x)
    
    model  = Model(input, x)
    
    return model


def cba(inputs, filters, kernel_size, strides):
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


# [2]Q関数をディープラーニングのネットワークをクラスとして定義
class QNetwork:
    def __init__(self, learning_rate=0.01):
        self.debug_log = True
        
        self.model = create_unet()
        
        self.optimizer = Adam(lr=learning_rate)  # 誤差を減らす学習方法はAdam
        self.model.compile(loss=huberloss, optimizer=self.optimizer)
        
        if self.debug_log == True:
            self.model.summary()

    # 重みの学習
    def replay(self, memory, batch_size, gamma, targetQN):
        inputs  = np.zeros((batch_size, 16, 16, 7))
        targets = np.zeros((batch_size, 16, 16, 1))
        mini_batch = memory.sample(batch_size)

        #for i, (state_b, linear_b, angle_b, reward_b, next_state_b) in enumerate(mini_batch):
        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            inputs[i:i + 1] = state_b
            target = reward_b

            #if not (next_state_b == np.zeros(state_b.shape)).all(axis=1): # 状態が全部ゼロじゃない場合
            if 1:
                # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
                retmainQs = self.model.predict(next_state_b)[0]    # (16, 16, 1)
                retmainQs = np.reshape(retmainQs, (16, 16))        # (16, 16)
                
                # 最大の報酬を返す行動を選択する
                next_action = np.unravel_index(np.argmax(retmainQs), retmainQs.shape)
                #print(next_action)
                #next_action = np.argmax(retmainQs)      # 最大の報酬を返す行動を選択する
                
                targetQs = targetQN.model.predict(next_state_b)[0] # (16, 16, 1)
                targetQs = np.reshape(targetQs, (16, 16))          # (16, 16, 1)
                
                target = reward_b + gamma * targetQs[next_action[0]][next_action[1]]
                #target = reward_b + gamma * targetQN.model.predict(next_state_b)[0][next_action]

            targets[i] = self.model.predict(state_b)               # Qネットワークの出力
            #targets[i][action_b] = target                         # 教師信号
            targets[i][action_b[0]][action_b[1]] = target          # 教師信号
            #print('**************************************' , i, targets[i].shape, action_b, target, targets[i])

        # shiglayさんよりアドバイスいただき、for文の外へ修正しました
        self.model.fit(inputs, targets, epochs=1, verbose=0)  # 初回は時間がかかる epochsは訓練データの反復回数、verbose=0は表示なしの設定


# [3]Experience ReplayとFixed Target Q-Networkを実現するメモリクラス
class Memory:
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.reset()
        #self.buffer = deque(maxlen=max_size)

    def reset(self):
        self.buffer = deque(maxlen=self.max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]

    def len(self):
        return len(self.buffer)


def generateRandomDestination(ban):
    flag = True
    while flag:
        flag   = False
        action = np.array( [int(3+np.random.rand()*11), int(3+np.random.rand()*11)] )
        for a in ban:
            if a[0] == action[0] and a[1] == action[1] : flag = True
    return action


# [4]カートの状態に応じて、行動を決定するクラス
# アドバイスいただき、引数にtargetQNを使用していたのをmainQNに修正しました
class Actor:
    def __init__(self):
        self.debug_log = True

    def get_action(self, state, episode, mainQN):   # [C]ｔ＋１での行動を返す
        # 徐々に最適行動のみをとる、ε-greedy法
        epsilon = 0.001 + 0.9 / (1.0+episode)
        #print(epsilon)
        epsilon = 0.3
        
        # 移動禁止箇所
        ban = np.array( [ [4,8], [7,8], [7,7], [8,12], [8,9], [8,8], [8,7], [8,4], [9,9], [9,8], [12,8]  ] )
        
        if epsilon <= np.random.uniform(0, 1):
            retTargetQs = mainQN.model.predict(state)[0]    # (16, 16, 1)
            retTargetQs = np.reshape(retTargetQs, (16, 16)) # (16, 16, 1)
            action      = np.unravel_index(np.argmax(retTargetQs), retTargetQs.shape)
            action      = np.array(action)
            
            # 学習結果が移動禁止箇所だったらランダムを入れておく
            flag   = False
            for a in ban:
                if a[0] == action[0] and a[1] == action[1] : flag = True
            if flag or action[0] < 3 or action[1] < 3 or action[0] > 13 or action[1] > 13:
                action = generateRandomDestination(ban)
                print('Random')
            else:
                print('Learned')
        else:
            print('Random')
            # 移動禁止箇所以外へランダムに行動する
            action = generateRandomDestination(ban)

        return action


