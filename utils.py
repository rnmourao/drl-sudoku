import numpy as np


def one_hot_encoder(board, n_classes=9):
    data = []
    
    flatten = board.flatten()
    for b in flatten():
        dummy_pos = [0] * n_classes
        if b != 0:
            i = b - 1
            dummy_pos[i] = 1   # digit position
        data += dummy_pos
    return data

