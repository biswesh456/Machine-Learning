import numpy as np
import random

class Environment():
    def __init__(self, matrix_size_array):
        self.matrix_size_array = matrix_size_array
        self.total_actions = 4
        self.total_matrices = len(self.matrix_size_array) - 1

    def getTotalStates(self):
        return self.total_matrices ** 2

    def getTotalActions(self):
        return self.total_actions

    def convertMatrixToState(self, start_index, end_index):
        return start_index * self.total_matrices + end_index

    def convertStateToMatrix(self, state):
        start_index = state // self.total_matrices
        end_index = state % self.total_matrices
        return start_index, end_index

    def reset(self):
        start_index = 0
        end_index = self.total_matrices - 1
        current_state = self.convertMatrixToState(start_index, end_index)
        return current_state

    def performAction(self, state, action):
        start_index, end_index = self.convertStateToMatrix(state)
        if start_index == end_index:
            return state, 0, True

        if(action == 0):
            # action 0 means ABCDEFG = (AB)*[CDEFG]
            if (start_index + 1) == end_index:
                return state, float('-inf'), True

            reward = -1 * self.matrix_size_array[start_index] \
                        * self.matrix_size_array[start_index+1] \
                        * self.matrix_size_array[start_index+2]

            reward -= self.matrix_size_array[start_index] * \
                      self.matrix_size_array[start_index+2] * \
                      self.matrix_size_array[end_index+1]

            next_state = self.convertMatrixToState(start_index+2, end_index)
            return next_state, reward, False

        elif(action == 1):
            # action 1 means ABCDEFG = A*[BCDEFG]
            reward = -1 * self.matrix_size_array[start_index] \
                        * self.matrix_size_array[start_index+1] \
                        * self.matrix_size_array[end_index+1]

            next_state = self.convertMatrixToState(start_index+1, end_index)
            return next_state, reward, False

        elif(action == 2):
            # action 2 means ABCDEFG = [ABCDE]*(FG)
            if (start_index + 1) == end_index:
                return state, float('-inf'), True

            reward = -1 * self.matrix_size_array[end_index-1] \
                        * self.matrix_size_array[end_index] \
                        * self.matrix_size_array[end_index+1]

            reward -= self.matrix_size_array[start_index] * \
                      self.matrix_size_array[end_index-1] * \
                      self.matrix_size_array[end_index+1]

            next_state = self.convertMatrixToState(start_index, end_index-2)
            return next_state, reward, False

        else:
            # action 3 means ABCDEFG = [ABCDEF]*G
            reward = -1 * self.matrix_size_array[start_index] \
                        * self.matrix_size_array[end_index] \
                        * self.matrix_size_array[end_index+1]

            next_state = self.convertMatrixToState(start_index, end_index-1)
            return next_state, reward, False

def printFinalAnswer(action_list, lst):
    for a in range(len(action_list)):
        action = action_list[a]
        print("Step ", a + 1, ": ")
        if action == 0 :
            print("(", lst[0], lst[1], ") * (", lst[2:], ")")
            lst = lst[2:]
        elif action == 1:
            print("(", lst[0], ") * (", lst[1:], ")")
            lst = lst[1:]

        elif action == 2:
            print("(", lst[:-3], ") * (", lst[-3], lst[-2], ")")
            lst = lst[:-3]

        else:
            print("(", lst[:-2], ") * (", lst[-2], ")")
            lst = lst[:-2]

if __name__ == "__main__":
    env = Environment([40, 20, 30, 10, 30])
    total_actions  = env.getTotalActions()
    total_states = env.getTotalStates()
    Q = np.zeros([total_states, total_actions])

    reward_list = []
    num_of_episodes = 2000
    gamma = 0.95
    for episode in range(num_of_episodes):
        current_state = env.reset()
        # reducing the epsilon as the time passes
        epsilon = 1/(1+(episode/100))
        while True:
            # epsilon-greedy method
            rnd = random.uniform(0, 1)
            action = np.argmax(Q[current_state])

            if rnd <= epsilon:
                a = random.randint(0, total_actions-1)
                while a == action:
                    a = random.randint(0, total_actions-1)
                action = a

            next_state, reward, done = env.performAction(current_state, action)
            estimated_reward = reward + gamma * np.max(Q[next_state, :])
            Q[current_state, action] = (Q[current_state, action] + estimated_reward)/2

            # change the Q for state with (end, start) instead of (start, end)
            start_index, end_index = env.convertStateToMatrix(current_state)
            reversed_state = env.convertMatrixToState(end_index, start_index)
            Q[reversed_state, action] = Q[current_state, action]

            reward_list.append(reward)
            current_state = next_state

            if done:
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
        action = np.argmax(Q[state])
        state, _, end = env.performAction(state, action)
        if not end:
            print("Action : ", action)
            action_list.append(action)

    print("Finished run ...... ")

    print("\nFinal Answer -> ")
    printFinalAnswer(action_list, env.matrix_size_array)
