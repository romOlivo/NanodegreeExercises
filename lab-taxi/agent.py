import numpy as np
import random
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.current_epsilon = 1.0
        self.default_epsilon = 1.0
        self.step_size = 1
        self.alpha = 0.1
        self.gamma = 0.95

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if random.random() > self.current_epsilon: # select greedy action with probability epsilon
            return np.argmax(self.Q[state])
        else:                     # otherwise, select an action randomly
            return np.random.choice(self.nA)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if done:
            self.step_size += 0.5
            self.current_epsilon = self.default_epsilon / self.step_size
        self.Q[state][action] += self.alpha * (reward + self.gamma * max(self.Q[next_state]) - self.Q[state][action])   
        