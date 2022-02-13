import numpy as np
import gym
from gym import spaces
from utils import *

class SudokuEnv(gym.Env):
    """
    Custom Environment that follows gym interface. 
    """

    def __init__(self):
        super(SudokuEnv, self).__init__()
        self.DIGITS = np.array((1, 2, 3, 4, 5, 6, 7, 8, 9))
        self.shape = np.array((9, 9))
        self.box_size = np.array((3, 3))
        self.row_size = 9
        self.fixed = np.zeros(self.shape) 

        self.observation_size = np.multiply(*self.shape) * self.DIGITS.size + \
                                self.fixed.size
        self.observation_space = spaces.MultiDiscrete([2] * self.observation_size)

        # the action space is the agent's move in the board
        self.action_space = spaces.Discrete(self.observation_size - self.fixed.size)


    def add_players(self, players):
        self.players = players


    def reset(self, fixed_numbers=dict()):
        """
        Important: the observation must be a numpy array
        :return: (np.array) 
        """
        board = np.zeros(self.shape)
        size = self.DIGITS.size
    
        # flat board and convert positions to dummy variables
        data = one_hot_encoder(board, size)

        # fill the the 2D-array with the fixed numbers
        for position in fixed_numbers.keys():
            board[position] = fixed_numbers[position]
        self.fixed = board.copy()

        mask = np.where(self.fixed.flatten() > 0, 1, 0).tolist()
        data += mask

        return np.array(data).astype(np.float32)


    def step(self, state, action):
        done = False
        outcome, next_state, _ = self.check_outcome()
        if outcome:
            done = True
        reward = self.calculate_reward(outcome)
        return outcome, next_state, reward, done


    def check_outcome(self, action):
        # prepare next_state
        mask = np.where(self.fixed.flatten() > 0, 1, 0).tolist()
        next_state = action.tolist() + mask
        
        # row or box coordinates where the mistake happened
        mistake = []

        # convert data to board
        flatten = [0] * self.action_space
        for pos in action.reshape((-1, 9)):
            ones = np.where(pos == 1)
            if len(ones) > 1:    # marked more than a number in a position
                return "mistake", next_state, mistake
            elif len(ones) == 0: # left empty
                return "", next_state, mistake
            i = ones[0]
            v = self.DIGITS[i]
            flatten.append(v)
        board = np.array(flatten).reshape(self.shape)
        
        # check fixed
        fixed_pos = np.where(self.fixed != 0)
        for pos in fixed_pos:
            if board[pos] != self.fixed[pos]:
                return "fixed", next_state, mistake

        # check rows
        for row in board:
            row_set = set(row)
            if len(row_set) < self.row_size:
                return "mistake", next_state, mistake

        # check columns
        for col in board.transpose():
            col_set = set(col)
            if len(col_set) < self.row_size:
                return "mistake", next_state, mistake

        # check boxes
        row_box_size = self.box_size[0]
        col_box_size = self.box_size[1]
        for i in np.arange(self.row_size, step=row_box_size):
            for j in np.arange(self.row_size, step=col_box_size):
                box = board[i:i+row_box_size, j:j+col_box_size].flatten()
                box_set = set(box)
                if len(box_set) < self.row_size:
                    return "mistake", next_state, mistake    

        return "win", next_state, mistake


    def calculate_reward(self, outcome):
        if outcome == "win":
            reward = 1
        elif outcome == "fixed": # changed a fixed number
            reward = -100
        elif outcome == "mistake": # broke rule
            reward = -10
        else: # some positions are still empty
            reward = -1
        return reward


    def render(self, state, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        print("board")


    def close(self):
        pass
