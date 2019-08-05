# -*- coding: utf-8 -*-

# _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
# _/  強化学習DQN (Deep Q Network)
# _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/

import numpy as np

from collections import deque
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model

# 次の行動を決める
def action_select(action):
    if action == 0 : linear = -0.2; angle = -1.0
    if action == 1 : linear = -0.2; angle =  0.0
    if action == 2 : linear = -0.2; angle =  1.0
    if action == 3 : linear =  0.0; angle = -1.0
    if action == 4 : linear =  0.0; angle =  0.0
    if action == 5 : linear =  0.0; angle =  1.0
    if action == 6 : linear =  0.2; angle = -1.0
    if action == 7 : linear =  0.2; angle =  0.0
    if action == 8 : linear =  0.2; angle =  1.0
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


# [2]Q関数をディープラーニングのネットワークをクラスとして定義
class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=28, action_size=9, hidden_size=10):
        self.state_size  = state_size
        self.action_size = action_size
        self.model = Sequential()
        self.model.add(Dense(hidden_size, activation='relu', input_dim=state_size))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))
        self.optimizer = Adam(lr=learning_rate)  # 誤差を減らす学習方法はAdam
        # self.model.compile(loss='mse', optimizer=self.optimizer)
        self.model.compile(loss=huberloss, optimizer=self.optimizer)
        self.model.summary()

    # 重みの学習
    def replay(self, memory, batch_size, gamma, targetQN):
        inputs  = np.zeros((batch_size,  self.state_size))
        targets = np.zeros((batch_size, self.action_size))
        mini_batch = memory.sample(batch_size)

        #for i, (state_b, linear_b, angle_b, reward_b, next_state_b) in enumerate(mini_batch):
        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            inputs[i:i + 1] = state_b
            target = reward_b

            if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
                retmainQs = self.model.predict(next_state_b)[0]
                next_action = np.argmax(retmainQs)      # 最大の報酬を返す行動を選択する
                target = reward_b + gamma * targetQN.model.predict(next_state_b)[0][next_action]

            targets[i] = self.model.predict(state_b)    # Qネットワークの出力
            targets[i][action_b] = target               # 教師信号

        # shiglayさんよりアドバイスいただき、for文の外へ修正しました
        self.model.fit(inputs, targets, epochs=1, verbose=0)  # epochsは訓練データの反復回数、verbose=0は表示なしの設定


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


# [4]カートの状態に応じて、行動を決定するクラス
# アドバイスいただき、引数にtargetQNを使用していたのをmainQNに修正しました
class Actor:

    def get_action(self, state, episode, mainQN):   # [C]ｔ＋１での行動を返す
        # 徐々に最適行動のみをとる、ε-greedy法
        epsilon = 0.001 + 0.9 / (1.0+episode)
        #print(epsilon)
        epsilon = 0.2

        if epsilon <= np.random.uniform(0, 1):
            retTargetQs   = mainQN.model.predict(state)[0]
            action        = np.argmax(retTargetQs)  # 最大の報酬を返す行動を選択する
        else:
            action = int(np.random.rand()*9)        # ランダムに行動する
        linear, angle = action_select(action)

        return action, linear, angle


