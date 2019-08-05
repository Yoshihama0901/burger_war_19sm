# -*- coding: utf-8 -*-

# _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
# _/  �����w�KDQN (Deep Q Network)
# _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/

import numpy as np

from collections import deque
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model

# ���̍s�������߂�
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

# [1]�����֐��̒�`
# �����֐���huber�֐����g�p���܂� �Q�lhttps://github.com/jaara/AI-blog/blob/master/CartPole-DQN.py
def huberloss(y_true, y_pred):
    err = y_true - y_pred
    cond = K.abs(err) < 1.0
    L2 = 0.5 * K.square(err)
    L1 = (K.abs(err) - 0.5)
    loss = tf.where(cond, L2, L1)  # Keras does not cover where function in tensorflow :-(
    return K.mean(loss)


# [2]Q�֐����f�B�[�v���[�j���O�̃l�b�g���[�N���N���X�Ƃ��Ē�`
class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=28, action_size=9, hidden_size=10):
        self.state_size  = state_size
        self.action_size = action_size
        self.model = Sequential()
        self.model.add(Dense(hidden_size, activation='relu', input_dim=state_size))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))
        self.optimizer = Adam(lr=learning_rate)  # �덷�����炷�w�K���@��Adam
        # self.model.compile(loss='mse', optimizer=self.optimizer)
        self.model.compile(loss=huberloss, optimizer=self.optimizer)
        self.model.summary()

    # �d�݂̊w�K
    def replay(self, memory, batch_size, gamma, targetQN):
        inputs  = np.zeros((batch_size,  self.state_size))
        targets = np.zeros((batch_size, self.action_size))
        mini_batch = memory.sample(batch_size)

        #for i, (state_b, linear_b, angle_b, reward_b, next_state_b) in enumerate(mini_batch):
        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            inputs[i:i + 1] = state_b
            target = reward_b

            if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                # ���l�v�Z�iDDQN�ɂ��Ή��ł���悤�ɁA�s�������Q�l�b�g���[�N�Ɖ��l�ϐ���Q�l�b�g���[�N�͕����j
                retmainQs = self.model.predict(next_state_b)[0]
                next_action = np.argmax(retmainQs)      # �ő�̕�V��Ԃ��s����I������
                target = reward_b + gamma * targetQN.model.predict(next_state_b)[0][next_action]

            targets[i] = self.model.predict(state_b)    # Q�l�b�g���[�N�̏o��
            targets[i][action_b] = target               # ���t�M��

        # shiglay������A�h�o�C�X���������Afor���̊O�֏C�����܂���
        self.model.fit(inputs, targets, epochs=1, verbose=0)  # epochs�͌P���f�[�^�̔����񐔁Averbose=0�͕\���Ȃ��̐ݒ�


# [3]Experience Replay��Fixed Target Q-Network���������郁�����N���X
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


# [4]�J�[�g�̏�Ԃɉ����āA�s�������肷��N���X
# �A�h�o�C�X���������A������targetQN���g�p���Ă����̂�mainQN�ɏC�����܂���
class Actor:

    def get_action(self, state, episode, mainQN):   # [C]���{�P�ł̍s����Ԃ�
        # ���X�ɍœK�s���݂̂��Ƃ�A��-greedy�@
        epsilon = 0.001 + 0.9 / (1.0+episode)
        #print(epsilon)
        epsilon = 0.2

        if epsilon <= np.random.uniform(0, 1):
            retTargetQs   = mainQN.model.predict(state)[0]
            action        = np.argmax(retTargetQs)  # �ő�̕�V��Ԃ��s����I������
        else:
            action = int(np.random.rand()*9)        # �����_���ɍs������
        linear, angle = action_select(action)

        return action, linear, angle


