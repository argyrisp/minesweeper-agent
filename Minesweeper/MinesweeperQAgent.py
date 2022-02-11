from pynput.mouse import Button
from pynput.keyboard import Key, Listener
from copy import deepcopy

import pynput
import time
import keyboard
import matplotlib.pyplot as pplot
import mss
import random as rand
import numpy as np
import os
import cv2
import pickle
import sys


mouse = pynput.mouse.Controller()
keyb = pynput.keyboard.Controller()


DISCOUNT = .95
a = .75

epsilon = 1
EPSILON_DECAY = 0.99995
MIN_EPSILON = 0.001

EPISODES = 500
load_flag = True
predict = True


def cell_recognition(scs, x, y):
    grey = [192, 192, 192]
    white = [255, 255, 255]
    black = [0, 0, 0]
    blue = [0, 0, 255]
    green = [0, 128, 0]
    red = [255, 0, 0]
    deep_blue = [0, 0, 128]
    deep_red = [128, 0, 0]
    cyan = [0, 128, 128]
    #print(scs.pixel((x) * 16 + 8, (y) * 16 + 8))
    if list(scs.pixel((x) * 16 + 8, (y) * 16 + 8)) == grey and list(
            scs.pixel((x) * 16 + 0, (y) * 16 + 0)) == white:
        # print("unidentified cell")
        return -1
    elif list(scs.pixel((x) * 16 + 8, (y) * 16 + 8)) == grey:
        # print("open grey cell")
        return 0
    elif list(scs.pixel((x) * 16 + 8, (y) * 16 + 8)) == black and list(scs.pixel((x) * 16 + 2, (y) * 16 + 8)) == black:
        # print("black cell")
        return -10
    elif list(scs.pixel((x) * 16 + 8, (y) * 16 + 8)) == blue:
        # print("blue : 1")
        return 1
    elif list(scs.pixel((x) * 16 + 8, (y) * 16 + 8)) == green:
        # print("green : 2")
        return 2
    elif list(scs.pixel((x) * 16 + 8, (y) * 16 + 8)) == red:
        # print("red : 3")
        return 3
    elif list(scs.pixel((x) * 16 + 8, (y) * 16 + 8)) == deep_blue:
        # print("deep blue : 4")
        return 4
    elif list(scs.pixel((x) * 16 + 8, (y) * 16 + 8)) == deep_red:
        # print("deep red : 5")
        return 5
    elif list(scs.pixel((x) * 16 + 8, (y) * 16 + 8)) == cyan:
        # print("cyan : 6")
        return 6
    else:
        print(scs.pixel((x) * 16 + 8, (y) * 16 + 8))
        return None


def board_score(scs):
    score = 0
    for i in range(0, 9):
        for j in range(0, 9):
            if cell_recognition(scs, i, j) != -1 and cell_recognition(scs, i, j) != -10:
                score += 1
    return score


def rand_action():
    mouse.position = (15 + 16 * rand.randint(0, 8) + 8, 105 + 16 * rand.randint(0, 8) + 8)
    # mouse.position = (15 + 16 * 0 + 8, 105 + 16 * 0 + 8)
    time.sleep(.05)
    mouse.click(Button.left, 2)


def state_conversion_array(scs):
    cell_list = []
    for m in range(0, 3):
        for n in range(0, 3):
            # temp.append(rgb2gray(scs.pixel(n, m)))
            cell = cell_recognition(scs, n, m)
            cell_list.append(cell)
    return cell_list


def reset_game():
    # Set active window
    mouse.position = (123, 40)
    time.sleep(.1)
    mouse.click(Button.left, 1)

    keyb.press(Key.f2)
    # time.sleep(.0005)
    keyb.release(Key.f2)

    rand_action()


def is_valid_action(scs, action):
    (j, i) = divmod(action, 3)
    # print(i, j)
    if i > 2 or j > 2:
        print("invalid action")
        return
    if cell_recognition(scs, i, j) != -1:
        return False
    else:
        return True


def hit_cell(action, idx):
    (j, i) = divmod(action, 3)
    (y, x) = divmod(idx, 7)
    if i > 2 or j > 2:
        print("invalid action")
        return
    if x > 7 or y > 7:
        print("invalid idx")
    # print(x, y)
    # print(i, j)

    mouse.position = (15 + 16 * i + 8 + x*16, 105 + 16 * j + 8 + y*16)
    time.sleep(.05)
    mouse.click(Button.left, 2)


def release_list(some_list):
    del some_list[:]
    del some_list


def clear_high_score():
    keyb.press(Key.enter)
    time.sleep(.05)
    keyb.release(Key.enter)
    time.sleep(.05)

    keyb.press(Key.enter)
    time.sleep(.05)
    keyb.release(Key.enter)
    time.sleep(.05)


def rgb2gray(rgb):
    return np.dot(rgb, [0.2989, 0.5870, 0.1140]) / 255


def state_conversion_array_2(scs):
    l1 = []
    for m in range(0, 9*16):
        l2 = []
        for n in range(0, 9*16):
            l2.append(rgb2gray(scs.pixel(n, m)))
        l1.append(l2)
    return l1


def board_score(scs):
    score = 0
    for i in range(0, 9):
        for j in range(0, 9):
            if cell_recognition(scs, i, j) == 1 or cell_recognition(scs, i, j) == 2:
                score += 1
    return score


if load_flag:
    f = open("Qtable19000.pkl", "rb")
    Qtable = pickle.load(f)
else:
    Qtable = {}

f = open("winrecordlist.txt", "r")
winrecordlist = f.read()
f.close()

# reset_game()

games_won = 0
game = 0
# epsilon = EPSILON_DECAY ** 19_000
epsilon = MIN_EPSILON
while game < 50_001:
    reset_game()
    game += 1
    done = False
    if keyboard.is_pressed('q'):
        loop = False
        break
    if game % EPISODES == 0:
        print("Game ", game, "-> won: ", games_won)
        print("length of table: ", len(Qtable))
        print("memory of table: ", sys.getsizeof(Qtable))
        f = open("Qtable"+str(game)+".pkl", "wb")
        pickle.dump(Qtable, f)
        f.close()
    # scs = mss.mss().grab({"top": 105, "left": 15, "width": 9*16, "height": 9*16})
    while not done:
        if keyboard.is_pressed('q'):
            loop = False
            break
        best_actions = []
        best_qs = []
        if np.random.rand() > epsilon or predict:

            for y in range(0, 7):
                for x in range(0, 7):
                    qs = []
                    # init_list = [float(0)] * 9
                    scs = mss.mss().grab({"top": 105 + y*16, "left": 15 + x*16, "width": 3 * 16, "height": 3 * 16})
                    cv2.imshow('win', np.array(scs))
                    state = state_conversion_array(scs)
                    # print(temp_list)
                    if -1 not in set(state):
                        qs = [float("-inf")] * 9
                    # states.append(temp_list)
                    else:
                        if str(state) not in Qtable:
                            Qtable[str(state)] = [0.0] * 9
                            # qs = deepcopy(init_list)
                            for action in range(0, 9):
                                # if not is_valid_action(scs, action):
                                if state[action] != -1:
                                    Qtable[str(state)][action] = float("-inf")
                            qs = deepcopy(Qtable[str(state)])
                        else:
                            qs = deepcopy(Qtable[str(state)])
                    # print(qs)
                    best_action = np.argmax(qs)
                    best_q = np.max(qs)
                    # print("act: ", best_action)
                    # print("q: ", best_q)
                    best_actions.append(best_action)
                    best_qs.append(best_q)

                    # for i in Qtable.items():
                    #     print(i)

            # print(best_actions)
            # print(best_qs)

            idx = int(np.argmax(best_qs))
            action = best_actions[idx]
            # print(idx)
            # print(action)
            (y, x) = divmod(idx, 7)
            scs = mss.mss().grab({"top": 105 + y*16, "left": 15 + x*16, "width": 3 * 16, "height": 3 * 16})
            # cv2.imshow('win', np.array(scs))
            state = state_conversion_array(scs)
            hit_cell(action, idx)
            (y, x) = divmod(idx, 7)
        else:
            while True:
                x = rand.randint(0, 6)
                y = rand.randint(0, 6)
                scs = mss.mss().grab({"top": 105 + y*16, "left": 15 + x*16, "width": 3 * 16, "height": 3 * 16})
                # cv2.imshow('win', np.array(scs))
                state = state_conversion_array(scs)
                if -1 in set(state):
                    break
            qs = [0.0] * 9
            if str(state) not in Qtable:
                for u in range(0, 9):
                    if state[u] != -1:
                        qs[u] = float("-inf")
                Qtable[str(state)] = qs
            else:
                qs = Qtable[str(state)]
            # print(state)
            action = np.argmax(qs)
            (j, i) = divmod(int(action), 3)
            # print(x, y)
            # print(i, j)
            mouse.position = (15 + 16 * i + 8 + x * 16, 105 + 16 * j + 8 + y * 16)
            time.sleep(.05)
            mouse.click(Button.left, 2)

        scs = mss.mss().grab({"top": 105, "left": 15, "width": 9 * 16, "height": 9 * 16})
        winrecord_check = np.array(state_conversion_array_2(scs))
        if str(winrecord_check) == winrecordlist:
            clear_high_score()

        release_list(best_actions)
        release_list(best_qs)
        scs = mss.mss().grab({"top": 105 + y*16, "left": 15 + x*16, "width": 3 * 16, "height": 3 * 16})
        # cv2.imshow('win2', np.array(scs))
        new_state = state_conversion_array(scs)
        if -10 in set(new_state):
            done = True
            reward = -1
            Qtable[str(state)][int(action)] = Qtable[str(state)][int(action)] + a*(reward - Qtable[str(state)][int(action)])
        elif board_score(mss.mss().grab({"top": 105, "left": 15, "width": 9 * 16, "height": 9 * 16})) == (9*9) - 10:
            done = True
            games_won += 1
            reward = 1
            Qtable[str(state)][int(action)] = Qtable[str(state)][int(action)] + a * (
                        reward - Qtable[str(state)][int(action)])
        else:
            reward = 1
            if str(new_state) not in Qtable:
                Qtable[str(new_state)] = [0] * 9
            max_q = np.max(Qtable[str(new_state)])
            Qtable[str(state)][int(action)] = Qtable[str(state)][int(action)] + a * (reward + (DISCOUNT*max_q) - Qtable[str(state)][int(action)])

    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)


# print("debug")
# for x in Qtable.items():
    # print(x)

