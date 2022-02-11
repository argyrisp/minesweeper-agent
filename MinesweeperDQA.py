from pynput.mouse import Button
from pynput.keyboard import Key, Listener
from PIL import Image

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import tensorflow as tf
from _collections import deque

import pynput
import time
import keyboard
import matplotlib.pyplot as pplot
import mss
import random as rand
import numpy as np
import os
import cv2

REPLAY_MEMORY_SIZE = 10_000
MIN_REPLAY_MEMORY_SIZE = 500
MINI_BATCH_SIZE = 32
MODEL_NAME = "256x2"
DISCOUNT = 0.98
UPDATE_TARGET_EVERY = 5
MIN_REWARD = -200

EPISODES = 200

epsilon = 1
EPSILON_DECAY = 0.99995
MIN_EPSILON = 0.001


class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


def create_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=(9*16, 9*16, 1)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(32))

    model.add(Dense(81, activation="linear"))
    model.compile(loss="mse", optimizer=Adam(lr=.001), metrics=['accuracy'])
    return model


class DQNAgent:
    def __init__(self):
        self.model = create_model()

        self.target_model = create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")

        self.target_update_counter = 0

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

    def train(self, terminal_state):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        minibatch = rand.sample(self.replay_memory, MINI_BATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max([index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X), np.array(y), batch_size=MINI_BATCH_SIZE, verbose=0, shuffle=False,
                       callbacks=[self.tensorboard] if terminal_state else None)

        # updating to determine if we want to update predict model yet
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


def rgb2gray(rgb):
    return np.dot(rgb, [0.2989, 0.5870, 0.1140]) / 255


def cell_recognition(scs, x, y):
    grey = [192, 192, 192]
    white = [255, 255, 255]
    black = [0, 0, 0]
    if list(scs.pixel((x) * 16 + 8, (y) * 16 + 8)) == grey and list(
            scs.pixel((x) * 16 + 0, (y) * 16 + 0)) == white:
        # print("unidentified cell")
        return 0
    elif list(scs.pixel((x) * 16 + 8, (y) * 16 + 8)) == grey:
        # print("open grey cell")
        return 1
    elif list(scs.pixel((x) * 16 + 8, (y) * 16 + 8)) == black:
        # print("black cell")
        return 3
    else:
        # print("color")
        return 2


def board_score(scs):
    score = 0
    for i in range(0, 9):
        for j in range(0, 9):
            if cell_recognition(scs, i, j) == 1 or cell_recognition(scs, i, j) == 2:
                score += 1
    return score


def state_conversion_array(scs):
    init_list = []
    for m in range(0, 9*16):
        temp = []
        for n in range(0, 9*16):
            temp.append(rgb2gray(scs.pixel(n, m)))
        init_list.append(temp)
    return init_list


def hit_cell(action):
    (j, i) = divmod(action, 9)
    if i > 8 or j > 8:
        print("invalid action")
        return
    mouse.position = (12 + 16 * i + 8, 98 + 16 * j + 8)
    time.sleep(.05)
    mouse.click(Button.left, 2)
    return i, j


def rand_action():
    mouse.position = (12 + 16 * rand.randint(0, 8) + 8, 98 + 16 * rand.randint(0, 8) + 8)
    # mouse.position = (15 + 16 * 0 + 8, 105 + 16 * 0 + 8)
    time.sleep(.05)
    mouse.click(Button.left, 2)


def clear_high_score():
    keyb.press(Key.enter)
    time.sleep(.05)
    keyb.release(Key.enter)
    time.sleep(.05)

    keyb.press(Key.enter)
    time.sleep(.05)
    keyb.release(Key.enter)
    time.sleep(.05)


def reset_game():

    keyb.press(Key.f2)
    # time.sleep(.0005)
    keyb.release(Key.f2)

    rand_action()


mouse = pynput.mouse.Controller()
keyb = pynput.keyboard.Controller()

mouse.position = (110, 14)
time.sleep(.1)
mouse.click(Button.left, 1)

# Screenshot, initial state
scs = mss.mss().grab({"top": 98, "left": 12, "width": 9*16, "height": 9*16})
state = np.array(state_conversion_array(scs=scs))  # input
state = state.reshape(*state.shape, -1)
# img = Image.fromarray(state)
# pplot.imshow(sct.pixels[0])

f = open("winrecordlist.txt", "r")
winrecordlist = f.read()
f.close()

agent = DQNAgent()
games_won = 0
game = 0
max_reward = 0
sum_reward = 0
temp_score = 0
while True:
    game += 1
    if game == 10000:
        break
    if keyboard.is_pressed('q'):
        loop = False
        break
    new_state_flag = False
    done = False
    # print("start game: ", game)
    agent.tensorboard.step = game
    step = 1
    if game % EPISODES == 0:
        print("GAME: ", game, ", max: ", max_reward, ", avg:", sum_reward/EPISODES)
        sum_reward = 0
        max_reward = 0
        agent.model.save('sweepmine'+str(game))
    while not done:
        reward = 1
        if keyboard.is_pressed('q'):
            loop = False
            break
        # cv2.imshow('win', state)
        ########################################################################################################
        # Enter learning here
        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(state))
            # print("chose action: ", action)
        else:
            action = rand.randint(0, 80)
        # action = rand.randint(0, 80)
        (j, i) = divmod(action, 9)
        if cell_recognition(scs=scs, x=i, y=j) != 0:
            reward = -1
        # take action
        hit_cell(action)
        ########################################################################################################
        scs = mss.mss().grab({"top": 98, "left": 12, "width": 9 * 16, "height": 9 * 16})
        new_state = np.array(state_conversion_array(scs))
        if str(new_state) == winrecordlist:
            clear_high_score()
            scs = mss.mss().grab({"top": 98, "left": 12, "width": 9 * 16, "height": 9 * 16})
            new_state = np.array(state_conversion_array(scs))
        # cv2.imshow('win2', new_state)
        new_state = new_state.reshape(*new_state.shape, -1)
        temp_score = board_score(scs)
        if temp_score > max_reward:
            max_reward = temp_score
        if cell_recognition(scs=scs, x=i, y=j) == 3:
            done = True
            reward = -10
            # print("black cell")
            # agent.update_replay_memory((np.array(state), action, reward, np.array(new_state), done))
            # agent.train(done)
            # step += 1
        else:
            reward *= 1
        if temp_score == (9*9)-10 and not done:
            # won
            done = True
            print("won game: ", game)
            games_won += 1
        agent.update_replay_memory((np.array(state), action, reward, np.array(new_state), done))
        agent.train(done)
        step += 1
        state = new_state
        # print('score: ', temp_score)
        # print('reward: ', reward)

    reset_game()
    sum_reward += temp_score
    scs = mss.mss().grab({"top": 98, "left": 12, "width": 9 * 16, "height": 9 * 16})
    state = np.array(state_conversion_array(scs=scs))  # input
    # cv2.imshow('win', state)
    state = state.reshape(*state.shape, -1)

    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
    # print(epsilon)
