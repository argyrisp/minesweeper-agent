from pynput.mouse import Button
from pynput.keyboard import Key, Listener

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

REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 100
MINI_BATCH_SIZE = 64
MODEL_NAME = "256x2"
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5
MIN_REWARD = -200

EPISODES = 200

epsilon = 1
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

mouse = pynput.mouse.Controller()
keyb = pynput.keyboard.Controller()
loop = True


# Set active window
mouse.position = (195, 20)
time.sleep(.1)
mouse.click(Button.left, 1)

if not os.path.isdir('models'):
    os.makedirs('models')

# Screenshot
sct = mss.mss().grab({"top": 101, "left": 15, "width": 30*16, "height": 16*16})
pplot.imshow(sct.pixels[0])


# Own Tensorboard class
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
    model.add(Conv2D(64, (3, 3), input_shape=(48, 48, 1)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(16))

    model.add(Dense(2, activation="linear"))
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
        #state = np.array(state).reshape(state.shape) / 255
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0] ############
        # return self.model.predict(state)[0]  ############

    def train(self, terminal_state):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        minibatch = rand.sample(self.replay_memory, MINI_BATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch]) / 255
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch]) / 255
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

        self.model.fit(np.array(X) / 255, np.array(y), batch_size=MINI_BATCH_SIZE, verbose=0, shuffle=False,
                       callbacks=[self.tensorboard] if terminal_state else None)

        # updating to determine if we want to update predict model yet
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


def rgb2gray(rgb):
    return np.dot(rgb, [0.2989, 0.5870, 0.1140])


def cell_recognition(scs, x, y):
    grey = [192, 192, 192]
    white = [255, 255, 255]
    black = [0, 0, 0]
    if list(scs.pixel((x - 1) * 16 + 8, (y - 1) * 16 + 8)) == grey and list(
            scs.pixel((x - 1) * 16 + 0, (y - 1) * 16 + 0)) == white:
        # print("unidentified cell")
        return 0
    elif list(scs.pixel((x - 1) * 16 + 8, (y - 1) * 16 + 8)) == grey:
        # print("open grey cell")
        return 1
    elif list(scs.pixel((x - 1) * 16 + 8, (y - 1) * 16 + 8)) == black:
        # print("black cell")
        return 3
    else:
        # print("color")
        return 2


def board_score(scs):
    score = 0
    for i in range(1, 30+1):
        for j in range(1, 16+1):
            if cell_recognition(scs, i, j) == 1 or cell_recognition(scs, i, j) == 2:
                score += 1
    return score


def state_conversion_array(scs, x, y):
    init_list = []
    lines = (3-1)/2  # lines above, below, left, right of center cell
    for m in range(0, 3*16):
        temp = []
        if y < lines+1 and m < lines*16:
            temp = [1.0] * (3*16)
            init_list.append(temp)
            continue
        if 16 - y < lines and (3-1)*16 <= m:
            temp = [1.0] * (3 * 16)
            init_list.append(temp)
            continue
        for n in range(0, 3*16):
            if x < lines+1 and n < lines*16:
                temp.append(1.0)
                continue
            if 30 - x < lines and (3-1)*16 <= n:
                temp.append(1.0)
                continue
            temp.append(rgb2gray(scs.pixel(n, m)))
        init_list.append(temp)
    return init_list


agent = DQNAgent()
game = 0
sum_reward = 0
max_reward = 0
games_won = 0
# Main loop
while True:
    if game == 5000:
        break
    game += 1
    agent.tensorboard.step = game
    game_reward = 0
    sum_reward += game_reward
    if game % EPISODES == 0:
        print("GAME: ", game, ", max: ", max_reward, ", avg:", sum_reward/EPISODES)
        sum_reward = 0
        max_reward = 0
        agent.model.save('sweepmine'+str(game))
    step = 1
    # print("====== Game: ", game, " ======")
    cell_black = False
    reward = 0
    action = -1
    if keyboard.is_pressed('q'):
        loop = False
        break
    new_state_flag = False
    while not cell_black:
        for j in range(1, 16+1):
            if cell_black:
                break
            if np.random.random() < epsilon:
                j = rand.randint(1, 16)
            if keyboard.is_pressed('q'):
                loop = False
                break
            for i in range(1, 30+1):
                if keyboard.is_pressed('q'):
                    loop = False
                    break

                scs = mss.mss().grab({"top": 101, "left": 15, "width": 30 * 16, "height": 16 * 16})
                if cell_recognition(scs=scs, x=i, y=j) != 0:
                    continue
                ########################################################################################################
                # Enter learning here
                state_ss = mss.mss().grab(
                    {"top": 101 + 16 * (j - 2), "left": 15 + 16 * (i - 2), "width": 16 * 3, "height": 16 * 3})
                state = np.array(state_conversion_array(scs=state_ss, x=i, y=j))  # input
                state = state.reshape(*state.shape, -1)
                if np.random.random() > epsilon:
                    action = np.argmax(agent.get_qs(state))
                    print("chose action: ", action)
                else:
                    action = np.random.randint(0, 2)

                if action == 0:
                    mouse.position = (15 + 16*(i-1) + 8, 101 + 16*(j-1) + 8)
                    time.sleep(.05)
                    mouse.click(Button.left, 2)
                # time.sleep(.05)
                ########################################################################################################
                    scs = mss.mss().grab({"top": 101, "left": 15, "width": 30 * 16, "height": 16 * 16})
                    new_state = np.array(state_conversion_array(scs, i, j))
                    new_state = new_state.reshape(*new_state.shape, -1)
                    if cell_recognition(scs=scs, x=i, y=j) == 3:
                        cell_black = True
                        reward = -10
                        agent.update_replay_memory((np.array(state), action, reward, np.array(new_state), cell_black))
                        agent.train(cell_black)
                        step += 1

                        temp_score = board_score(scs)
                        if temp_score > max_reward:
                            max_reward = temp_score
                        sum_reward += temp_score
                        break
                    else:
                        reward = 1
                else:
                    reward = 0
                    new_state = state
                temp_score = board_score(scs)
                if temp_score > max_reward:
                    max_reward = temp_score
                sum_reward += temp_score
                if temp_score == (30*16)-99:
                    # won
                    cell_black = True
                    print("won game: ", game)
                    games_won += 1
                    reward = 10
                agent.update_replay_memory((np.array(state), action, reward, np.array(new_state), cell_black))
                agent.train(cell_black)
                step += 1

    keyb.press(Key.f2)
    # time.sleep(.0005)
    keyb.release(Key.f2)

    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)






