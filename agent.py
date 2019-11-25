# -*- coding: utf-8 -*-

import gym
import random
import os.path
import time
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from collections import deque
from datetime import datetime
from keras import backend as K
from keras.optimizers import RMSprop
from keras.models import load_model
from keras.models import clone_model
from keras.models import Sequential
from keras.models import Model
from keras import layers
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dense


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Agent(object):
    def __init__(self):
        self.game = "Breakout-v0"
        self.train_dir = "saves"
        self.restore_file_path = "{0}/{1}.h5".format(self.train_dir, self.game)
        self.num_episode = 1000
        self.observe_step_num = 500
        self.epsilon_step_num = 10000
        self.refresh_target_model_num = 100
        self.replay_memory = 40000
        self.no_op_steps = 30
        self.regularizer_scale = 0.01
        self.batch_size = 32
        self.learning_rate = 0.00025
        self.init_epsilon = 1.0
        self.final_epsilon = 0.1
        self.gamma = 0.99
        self.resume = False
        self.render = True
        self.shape = (84, 84, 4)
        self.action_size = 3
        self.debug = False

    def set_game(self, game):
        self.game = game

    def train_dir(self, train_dir):
        self.train_dir = train_dir

    def set_restore_file_path(self, restore_file_path):
        self.restore_file_path = restore_file_path

    def num_episode(self, num_episode):
        self.num_episode = num_episode

    def observe_step_num(self, observe_step_num):
        self.observe_step_num = observe_step_num

    def set_epsilon_step_num(self, epsilon_step_num):
        self.epsilon_step_num = epsilon_step_num

    def set_refresh_target_model_num(self, refresh_target_model_num):
        self.refresh_target_model_num = refresh_target_model_num

    def set_replay_memory(self, replay_memory):
        self.replay_memory = replay_memory

    def set_no_op_steps(self, no_op_steps):
        self.no_op_steps = no_op_steps

    def set_regularizer_scale(self, regularizer_scale):
        self.regularizer_scale = regularizer_scale

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_init_epsilon(self, init_epsilon):
        self.init_epsilon = init_epsilon

    def set_final_epsilon(self, final_epsilon):
        self.final_epsilon = final_epsilon

    def set_gamma(self, gamma):
        self.gamma = gamma

    def set_resume(self, resume):
        self.resume = resume

    def set_render(self, render):
        self.render = render

    def set_shape(self, shape):
        self.shape = shape

    def set_action_size(self, action_size):
        self.action_size = action_size

    def set_debug(self, debug):
        self.debug = debug

    def pre_processing(self, observe):
        processed_observe = np.uint8(resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
        return processed_observe

    def huber_loss(self, y, q_value):
        error = K.abs(y - q_value)
        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
        return loss

    def def_model(self):
        frames_input = layers.Input(self.shape, name='frames')
        actions_input = layers.Input((self.action_size,), name='action_mask')
        normalized = layers.Lambda(lambda x: x / 255.0, name='normalization')(frames_input)
        conv_1 = layers.convolutional.Conv2D(
            16, (8, 8), strides=(4, 4), activation='relu'
        )(normalized)
        conv_2 = layers.convolutional.Conv2D(
            32, (4, 4), strides=(2, 2), activation='relu'
        )(conv_1)
        conv_flattened = layers.core.Flatten()(conv_2)
        hidden = layers.Dense(256, activation='relu')(conv_flattened)
        output = layers.Dense(self.action_size)(hidden)
        filtered_output = layers.Multiply(name='QValue')([output, actions_input])

        model = Model(inputs=[frames_input, actions_input], outputs=filtered_output)
        model.summary()
        optimizer = RMSprop(lr=self.learning_rate, rho=0.95, epsilon=0.01)
        model.compile(optimizer, loss=self.huber_loss)

        return model

    def get_action(self, history, epsilon, step, model):
        if np.random.rand() <= epsilon or step <= self.observe_step_num:
            return random.randrange(self.action_size)
        else:
            q_value = model.predict([history, np.ones(self.action_size).reshape(1, self.action_size)])
            return np.argmax(q_value[0])

    def store_memory(self, memory, history, action, reward, next_history, dead):
        memory.append((history, action, reward, next_history, dead))

    def get_one_hot(self, targets, nb_classes):
        return np.eye(nb_classes)[np.array(targets).reshape(-1)]

    def train_memory_batch(self, memory, model):
        mini_batch = random.sample(memory, self.batch_size)
        history = np.zeros((self.batch_size, self.shape[0], self.shape[1], self.shape[2]))
        next_history = np.zeros((self.batch_size, self.shape[0], self.shape[1], self.shape[2]))
        target = np.zeros((self.batch_size,))
        action = []
        reward = []
        dead = []

        for idx, val in enumerate(mini_batch):
            history[idx] = val[0]
            next_history[idx] = val[3]
            action.append(val[1])
            reward.append(val[2])
            dead.append(val[4])

        actions_mask = np.ones((self.batch_size, self.action_size))
        next_Q_values = model.predict([next_history, actions_mask])

        for i in range(self.batch_size):
            if dead[i]:
                target[i] = -1
            else:
                target[i] = reward[i] + self.gamma * np.amax(next_Q_values[i])

        action_one_hot = self.get_one_hot(action, self.action_size)
        target_one_hot = action_one_hot * target[:, None]

        h = model.fit([history, action_one_hot], target_one_hot, epochs=1, batch_size=self.batch_size, verbose=0)

        return h.history['loss'][0]

    def train(self, env):
        memory = deque(maxlen=self.replay_memory)
        episode_number = 0
        epsilon = self.init_epsilon
        epsilon_decay = (self.init_epsilon - self.final_epsilon) / self.epsilon_step_num
        global_step = 0

        if self.resume:
            model = load_model(self.restore_file_path)
            epsilon = self.final_epsilon
        else:
            model = self.def_model()

        model_target = clone_model(model)
        model_target.set_weights(model.get_weights())

        while episode_number < self.num_episode:

            done = False
            dead = False
            step, score, start_life = 0, 0, 5
            loss = 0.0
            observe = env.reset()

            for _ in range(random.randint(1, self.no_op_steps)):
                observe, score, done, info = env.step(1)

            if self.debug:
                print("Time Score: {0}".format(score))

            state = self.pre_processing(observe)
            history = np.stack((state, state, state, state), axis=2)
            history = np.reshape([history], (1, 84, 84, 4))

            while not done:
                if self.render:
                    env.render()
                    time.sleep(0.01)

                action = self.get_action(history, epsilon, global_step, model_target)

                if epsilon > self.final_epsilon and global_step > self.observe_step_num:
                    epsilon -= epsilon_decay

                observe, reward, done, info = env.step(action)
                next_state = self.pre_processing(observe)
                next_state = np.reshape([next_state], (1, 84, 84, 1))
                next_history = np.append(next_state, history[:, :, :, :3], axis=3)

                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']

                self.store_memory(memory, history, action, reward, next_history, dead)

                if global_step > self.observe_step_num:
                    loss = loss + self.train_memory_batch(memory, model)
                    if global_step % self.refresh_target_model_num == 0:
                        model_target.set_weights(model.get_weights())

                score += reward

                if dead:
                    dead = False
                else:
                    history = next_history

                global_step += 1
                step += 1

                if done:
                    if global_step <= self.observe_step_num:
                        state = "observe"
                    elif self.observe_step_num < global_step <= self.observe_step_num + self.epsilon_step_num:
                        state = "explore"
                    else:
                        state = "train"
                    print('state: {0}, episode: {1}, score: {2}'.format(
                        state, episode_number, score
                        ))

                    if episode_number % 100 == 0 or (episode_number + 1) == self.num_episode:
                        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
                        file_name = "{0}_model_{1}.h5".format(self.game, now)
                        model_path = os.path.join(self.train_dir, file_name)
                        model.save(model_path)

                    episode_number += 1

    def test(self, env):
        episode_number = 0
        epsilon = 0.001
        global_step = self.observe_step_num+1
        model = load_model(self.restore_file_path, custom_objects={'huber_loss': self.huber_loss})  # load model with customized loss func

        while episode_number < self.num_episode:

            done = False
            dead = False
            score, start_life = 0, 5
            observe = env.reset()

            observe, _, _, _ = env.step(1)
            state = self.pre_processing(observe)
            history = np.stack((state, state, state, state), axis=2)
            history = np.reshape([history], (1, 84, 84, 4))

            while not done:
                env.render()
                time.sleep(0.01)

                action = self.get_action(history, epsilon, global_step, model)

                observe, reward, done, info = env.step(action)
                next_state = self.pre_processing(observe)
                next_state = np.reshape([next_state], (1, 84, 84, 1))
                next_history = np.append(next_state, history[:, :, :, :3], axis=3)

                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']

                reward = np.clip(reward, -1., 1.)

                score += reward

                if dead:
                    dead = False
                else:
                    history = next_history

                global_step += 1

                if done:
                    episode_number += 1
                    print('episode: {0}, score: {1}'.format(episode_number, score))
