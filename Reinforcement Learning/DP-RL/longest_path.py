'''
Given a n*n matrix where all numbers are distinct, find the maximum length
path (starting from any cell) such that all cells along the path are in increasing
order with a difference of 1
'''

import numpy as np
import random

class Environment():
    def __init__(self, matrix):
        self.matrix = matrix
        (self.rows, self.columns) = self.matrix.shape
        self.total_actions = 4

    def getTotalStates(self):
        return self.rows * self.columns

    def getTotalActions(self):
        return self.total_actions

    def convertMatrixToState(self, row, col):
        return row * self.columns + col

    def convertStateToMatrix(self, state):
        row = state // self.columns
        col = state % self.columns
        return row, col

    def reset(self):
        row = random.randint(0, self.rows - 1)
        col = random.randint(0, self.columns - 1)
        current_state = self.convertMatrixToState(row, col)
        return current_state

    def performAction(self, state, action):
        row, col = self.convertStateToMatrix(state)

        if action == 0:
            # move up
            if row == 0:
                return state, float('-inf'), True

            elif self.matrix[row-1][col] - self.matrix[row][col] == 1:
                next_state = self.convertMatrixToState(row-1, col)
                return next_state, 1, False

            else:
                return state, 0, True

        elif action == 1:
            # move down
            if row == self.rows - 1:
                return state, float('-inf'), True

            elif self.matrix[row+1][col] - self.matrix[row][col] == 1:
                next_state = self.convertMatrixToState(row+1, col)
                return next_state, 1, False

            else:
                return state, 0, True

        elif action == 2:
            # move left
            if col == 0:
                return state, float('-inf'), True

            elif self.matrix[row][col-1] - self.matrix[row][col] == 1:
                next_state = self.convertMatrixToState(row, col-1)
                return next_state, 1, False

            else:
                return state, 0, True

        elif action == 3:
            # move right
            if col == self.columns - 1:
                return state, float('-inf'), True

            elif self.matrix[row][col+1] - self.matrix[row][col] == 1:
                next_state = self.convertMatrixToState(row, col+1)
                return next_state, 1, False

            else:
                return state, 0, True


if __name__ == "__main__":
    matrix = np.array([ [1, 2, 9],
                        [5, 3, 8],
                        [4, 6, 7] ], np.int32)

    print("Matrix \n", matrix)
    env = Environment(matrix)

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
    print("\nFinal Q")
    print(Q)

    max_state = (np.argmax(Q))//env.getTotalActions()
    r, c = env.convertStateToMatrix(max_state)
    print("\nAnswer : ")
    print("Row -> ", r+1)
    print("Column -> ", c+1)
