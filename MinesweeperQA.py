from pynput.mouse import Button
from pynput.keyboard import Key
from copy import deepcopy

import pynput
import time
from datetime import datetime
import keyboard
import mss
import random as rand
import numpy as np
import pickle
import sys


mouse = pynput.mouse.Controller()
keyb = pynput.keyboard.Controller()

table_name = "Qtable392000.pkl"

DISCOUNT = .95
a = .75


EPSILON_DECAY = 0.999975
EPSILON = 1
MIN_EPSILON = 0.001

EPISODES = 4000
load_flag = True
predict = True
GAMES_WON = 0


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
    # print(scs.pixel((x) * 16 + 8, (y) * 16 + 8))
    if (list(scs.pixel((x) * 16 + 5, (y) * 16 + 6)) == white and list(
            scs.pixel((x) * 16 + 7, (y) * 16 + 7)) == grey) or (list(scs.pixel((x) * 16 + 2, (y) * 16 + 0)) == white and
                                                                list(scs.pixel((x) * 16 + 3, (y) * 16 + 0)) == grey) or \
                                                                (list(scs.pixel((x) * 16 + 0, (y) * 16 + 2)) == white and
                                                                list(scs.pixel((x) * 16 + 0, (y) * 16 + 3)) == grey):
        # print("wall")
        return 100
    elif list(scs.pixel((x) * 16 + 8, (y) * 16 + 8)) == grey and list(
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
        # print(scs.pixel((x) * 16 + 8, (y) * 16 + 8))
        return None


def state_conversion_array(scs):
    cell_list = []
    for m in range(0, 5):
        temp_list = []
        for n in range(0, 5):
            temp_list.append(cell_recognition(scs, n, m))
        cell_list.append(temp_list)
    return cell_list


def board_score(scs):
    score = 0
    for i in range(0, 9):
        for j in range(0, 9):
            # print(cell_recognition(scs, i, j))
            if cell_recognition(scs, i, j) != -1 and cell_recognition(scs, i, j) != -10 and cell_recognition(scs, i, j) is not None:
                score += 1
    return score


def hit_cell(idx, idy):
    mouse.position = (12 + 16 * idx + 8, 98 + 16 * idy + 8)
    time.sleep(.05)
    mouse.click(Button.left, 2)


def rand_action():
    mouse.position = (12 + 16 * rand.randint(0, 8) + 8, 98 + 16 * rand.randint(0, 8) + 8)
    # mouse.position = (15 + 16 * 0 + 8, 105 + 16 * 0 + 8)
    time.sleep(.05)
    mouse.click(Button.left, 2)


def reset_game():
    # Set active window
    # mouse.position = (110, 14)
    # time.sleep(.1)
    # mouse.click(Button.left, 1)

    keyb.press(Key.f2)
    # time.sleep(.0005)
    keyb.release(Key.f2)

    rand_action()


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


def clear_high_score():
    # mouse.position = (73, 156)
    # time.sleep(.05)
    # mouse.click(Button.left, 2)
    # time.sleep(.05)

    keyb.press(Key.enter)
    time.sleep(.05)
    keyb.release(Key.enter)
    time.sleep(.05)

    keyb.press(Key.enter)
    time.sleep(.05)
    keyb.release(Key.enter)
    time.sleep(.05)


if load_flag:
    f = open(table_name, "rb")
    Qtable = pickle.load(f)
    f.close()
else:
    Qtable = {}


def q_learning(Qtable=Qtable, EPSILON=EPSILON, GAMES_WON=GAMES_WON):
    done = False
    while not done:
        if keyboard.is_pressed('q'):
            break

        idx = -1
        idy = -1
        # print(EPSILON)
        if np.random.rand() > EPSILON or predict:
            # print('prediction')

            states = []
            screenshots = []
            best_state = []
            best_action = -1
            best_q = float("-inf")
            xx = -1
            yy = -1
            for y in range(0, 3):
                for x in range(0, 3):
                    scs = mss.mss().grab({"top": 98 - 16 + y*3*16, "left": 12 - 16 + x*3*16, "width": 5 * 16, "height": 5 * 16})
                    # cv2.imshow('win', np.array(scs))
                    state = state_conversion_array(scs)

                    possible_actions = [state[1][1], state[1][2], state[1][3], state[2][1], state[2][2], state[2][3], state[3][1],
                                        state[3][2], state[3][3]]

                    if -1 not in set(possible_actions):
                        continue
                    # print(state)
                    # print(possible_actions)
                    states.append(state)
                    screenshots.append(scs)
                    if str(state) not in Qtable:
                        Qtable[str(state)] = [0.0] * 9
                        for u in range(0, 9):
                            if possible_actions[u] != -1:
                                Qtable[str(state)][u] = float("-inf")
                    action = np.argmax(Qtable[str(state)])
                    best_q_state = np.max(Qtable[str(state)])
                    # print(action)
                    # print(best_q_state)
                    if best_q_state > best_q:
                        best_q = best_q_state
                        best_action = action
                        (j, i) = divmod(int(action), 3)
                        idx = i + x*3
                        idy = j + y*3
                        best_state = state
                        xx = x
                        yy = y
                    # print("act: ", best_action)
                    # print("q: ", best_q)
            # print(best_q)
            # print(best_action)
            state = deepcopy(best_state)
            action = deepcopy(best_action)
            # hit_cell(idx, idy)

        else:
            while True:
                x = rand.randint(0, 2)
                y = rand.randint(0, 2)
                scs = mss.mss().grab({"top": 98 + (y * 16 * 3) - 16, "left": 12 + (x * 16 * 3) - 16, "width": 5 * 16, "height": 5 * 16})
                # cv2.imshow('win', np.array(scs))
                state = state_conversion_array(scs)
                possible_actions = [state[1][1], state[1][2], state[1][3], state[2][1], state[2][2], state[2][3],
                                    state[3][1],
                                    state[3][2], state[3][3]]

                if -1 in set(possible_actions):
                    break
            # Qtable[str(state)] = [0.0] * 9
            if str(state) not in Qtable:
                Qtable[str(state)] = [0.0] * 9
                for u in range(0, 9):
                    if possible_actions[u] != -1:
                        Qtable[str(state)][u] = float("-inf")
            action = rand.randint(0, 8)
            while possible_actions[action] != -1:
                action = rand.randint(0, 8)
            # best_action = np.argmax(Qtable[str(state)])
            (j, i) = divmod(int(action), 3)
            idx = (x*3) + i
            idy = (y*3) + j
            xx = x
            yy = y
            # print(state)
            # print(possible_actions)
            # print(Qtable[str(state)])
        # print(action, "=============")
        # print(xx)
        # print(yy)
        # print(state)
        hit_cell(idx, idy)
        scs = mss.mss().grab({"top": 98, "left": 12, "width": 9 * 16, "height": 9 * 16})
        if scs.pixel(20, 83) == (0, 120, 215):
            clear_high_score()

        scs = mss.mss().grab(
                {"top": 98 + (yy * 16 * 3) - 16, "left": 12 + (xx * 16 * 3) - 16, "width": 5 * 16, "height": 5 * 16})
        # cv2.imshow('win', np.array(scs))
        new_state = state_conversion_array(scs)
        new_state_3x3 = [new_state[1][1], new_state[1][2], new_state[1][3], new_state[2][1], new_state[2][2], new_state[2][3], new_state[3][1], new_state[3][2], new_state[3][3]]

        if board_score(mss.mss().grab({"top": 98, "left": 12, "width": 9 * 16, "height": 9 * 16})) == (9*9) - 10:
            done = True
            print('won')
            GAMES_WON += 1
            reward = 1
            Qtable[str(state)][int(action)] = Qtable[str(state)][int(action)] + a * (
                            reward - Qtable[str(state)][int(action)])
        elif -10 in set(new_state_3x3):
            done = True
            reward = -5
            Qtable[str(state)][int(action)] = Qtable[str(state)][int(action)] + a*(reward - Qtable[str(state)][int(action)])
        else:
            reward = 1
            if -1 not in set(new_state_3x3):
                Qtable[str(state)][int(action)] = Qtable[str(state)][int(action)] + a * (
                            reward - Qtable[str(state)][int(action)])
            else:
                if str(new_state) not in Qtable:
                    Qtable[str(new_state)] = [0.0] * 9
                    for u in range(0, 9):
                        if new_state_3x3[u] != -1:
                            Qtable[str(new_state)][u] = float("-inf")
                max_q = np.max(Qtable[str(new_state)])
                Qtable[str(state)][int(action)] = Qtable[str(state)][int(action)] + a * (reward + (DISCOUNT*max_q) - Qtable[str(state)][int(action)])
    if EPSILON > MIN_EPSILON:
        EPSILON *= EPSILON_DECAY
        EPSILON = max(MIN_EPSILON, EPSILON)
    return Qtable, EPSILON, GAMES_WON


mouse.position = (110, 14)
time.sleep(.1)
mouse.click(Button.left, 1)
game = 352_000
while game < 500_001:
    game += 1
    reset_game()
    if keyboard.is_pressed('q'):
        break
    if game % EPISODES == 0:
        print("======================================")
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time: ", current_time)
        print("Game ", game, "-> won: ", GAMES_WON)
        print("length of table: ", len(Qtable))
        print("memory of table: ", sys.getsizeof(Qtable), "\n")
        f = open("Qtable" + str(game) + ".pkl", "wb")
        pickle.dump(Qtable, f)
        f.close()
    Qtable, EPSILON, GAMES_WON = q_learning(Qtable, EPSILON, GAMES_WON)






