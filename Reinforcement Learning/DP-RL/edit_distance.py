'''
Given two strings str1 and str2 and operations(of replace, remove, insert) that
can performed on str1. Find minimum number of edits (operations) required to
convert ‘str1’ into ‘str2’
'''

import numpy as np
import random
import time

class Environment():
    def __init__(self, str1, str2):
        self.str1 = str1
        self.str2 = str2
        self.total_actions = 4

    def getTotalStates(self):
        return len(self.str1)*len(self.str2)

    def getTotalActions(self):
        return self.total_actions

    def convertMatrixToState(self, index1, index2):
        return index1 * len(self.str2) + index2

    def convertStateToMatrix(self, state):
        index1 = state // len(self.str2)
        index2 = state % len(self.str2)
        return index1, index2

    def reset(self):
        index1 = 0
        index2 = 0
        current_state = self.convertMatrixToState(index1, index2)
        return current_state

    def performAction(self, state, action):
        index1, index2 = self.convertStateToMatrix(state)

        if action == 0:
            # case where we don't do anything and move to next index of both strings
            if self.str1[index1] == self.str2[index2]:
                index1 += 1
                index2 += 1
                if index1 == len(self.str1):
                    return state, index2 - len(self.str2) - 1, True

                if index2 == len(self.str2):
                    return state, index1 - len(self.str1) - 1, True

                next_state = self.convertMatrixToState(index1, index2)
                return next_state, 1, False

            else:
                return state, float('-inf'), True

        elif action == 1:
            # remove
            index1 += 1
            if index1 == len(self.str1):
                return state, index2 - len(self.str2) - 1, True

            next_state = self.convertMatrixToState(index1, index2)
            return next_state, -1, False

        elif action == 2:
            # insert
            index2 += 1
            if index2 == len(self.str2):
                return state, index1 - len(self.str1) - 1, True

            next_state = self.convertMatrixToState(index1, index2)
            return next_state, -1, False

        else:
            # replace
            index1 += 1
            index2 += 1
            if index1 == len(self.str1):
                return state, index2 - len(self.str2) - 1, True

            if index2 == len(self.str2):
                return state, index1 - len(self.str1) - 1, True

            next_state = self.convertMatrixToState(index1, index2)
            return next_state, -1, False

if __name__ == "__main__":
    env = Environment("sunday", "saturday")

    total_actions  = env.getTotalActions()
    total_states = env.getTotalStates()
    Q = np.zeros([total_states, total_actions])

    reward_list = []
    num_of_episodes = 5000
    gamma = 1.0
    for episode in range(num_of_episodes):
        current_state = env.reset()
        # reducing the epsilon as the time passes
        epsilon = 1/(1+(episode/200))
        while True:
            # epsilon-greedy method
            rnd = random.uniform(0, 1)
            action = np.argmax(Q[current_state, :])

            if rnd <= epsilon:
                a = random.randint(0, total_actions-1)
                while a == action:
                    a = random.randint(0, total_actions-1)
                action = a

            next_state, reward, done = env.performAction(current_state, action)

            if not done:
                estimated_reward = reward + gamma * np.max(Q[next_state, :])
                Q[current_state, action] = (Q[current_state, action] + estimated_reward)/2
                reward_list.append(reward)
                current_state = next_state
            else:
                Q[current_state, action] = (Q[current_state, action] + reward)/2
                reward_list.append(reward)
                break

    #------------------------------Training Finished----------------------------
    print("Final Q")
    print(Q)

    # final run for answer
    print("\nFinal RUN ..... ")
    end = False
    state = env.reset()
    action_list = []

    while not end:
        print("State : ", state)
        action = np.argmax(Q[state, :])
        state, _, end = env.performAction(state, action)
        if not end:
            print("Action : ", action)
            action_list.append(action)

    print("Finished run ...... ")

    answer = 0
    for i in range(len(action_list)):
        if action_list[i] != 0:
            answer += 1

    print("\nFinal Answer -> ", answer)
