from tenuOS.util.constants import *

def is_valid_cell(r, q, board_size):
    return 0 <= r < board_size and 0 <= q < board_size

def is_connected_diagonal(r, q, board_size):
    return ((r + q) + 1 == board_size)

def print_state(state):
    """
    Prints the current state to console
    """
    print("\nstate as a grid: \n")
    for r in reversed(state):
        for q in r:
            print(q, end = ", ")
        print("")
    print("")

def opposite_colour(colour):
    if colour == RED:
        return BLUE
    elif colour == BLUE:
        return RED
    else:
        return EMPTY