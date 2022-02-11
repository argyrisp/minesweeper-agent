from pynput.mouse import Button
from pynput.keyboard import Key, Listener

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import tensorflow as tf
from _collections import deque
from keras.models import load_model

import pynput
import time
import keyboard
import matplotlib.pyplot as pplot
import mss
import random as rand
import numpy as np
import os

EPISODES = 200


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


model = load_model('sweepmine200')
game = 0
sum_reward = 0
max_reward = 0
games_won = 0
# Main loop
while game < 1001:
    game += 1
    game_reward = 0
    sum_reward += game_reward
    if game % EPISODES == 0:
        print("GAME: ", game, ", max: ", max_reward, ", avg:", sum_reward/EPISODES)
        sum_reward = 0
        max_reward = 0
    step = 1
    # print("====== Game: ", game, " ======")
    cell_black = False
    reward = 0
    action = -1
    if keyboard.is_pressed('q'):
        loop = False
        break
    while not cell_black:
        for j in range(1, 16+1):
            if cell_black:
                break
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
                action = np.argmax(model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0])

                if action == 0:
                    mouse.position = (15 + 16*(i-1) + 8, 101 + 16*(j-1) + 8)
                    time.sleep(.05)
                    mouse.click(Button.left, 2)
                # time.sleep(.05)
                ########################################################################################################
                    scs = mss.mss().grab({"top": 101, "left": 15, "width": 30 * 16, "height": 16 * 16})
                    if cell_recognition(scs=scs, x=i, y=j) == 3:
                        cell_black = True

                        temp_score = board_score(scs)
                        if temp_score > max_reward:
                            max_reward = temp_score
                        sum_reward += temp_score
                        break
                temp_score = board_score(scs)
                if temp_score == (30*16)-99:
                    # won
                    cell_black = True
                    print("won game: ", game)
                    games_won += 1
                    if temp_score > max_reward:
                        max_reward = temp_score
                    sum_reward += temp_score
                    break

    keyb.press(Key.f2)
    # time.sleep(.0005)
    keyb.release(Key.f2)
