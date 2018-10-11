"""Utils for visualization"""
import numpy as np


def make_board_rgb(env, color_bg):
    """ Get rgb color array from environment state. """
    board = env.current_game._board.board
    board_rgb = np.zeros((3, board.shape[0], board.shape[1]))

    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            rgb = np.array(color_bg[chr(board[i, j])]) / 1000
            board_rgb[:, i, j] = rgb

    return board_rgb
