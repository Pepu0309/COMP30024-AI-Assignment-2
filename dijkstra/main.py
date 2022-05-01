"""
COMP30024 Artificial Intelligence, Semester 1, 2022
Project Part A: Searching

This script contains the entry point to the program (the code in
`__main__.py` calls `main()`). Your solution starts here!
"""

import sys
import json

# If you want to separate your code into separate files, put them
# inside the `search` directory (like this one and `util.py`) and
# then import from them like this:
import util
import pathfinding
from tenuOS.enums import *
from tenuOS.player import SucessorState

def main():

    try:
        with open(sys.argv[1]) as file:
            data = json.load(file)
    except IndexError:
        print("usage: python3 -m search path/to/input.json", file=sys.stderr)
        sys.exit(1)

    successor_state = SuccesorState(state_from_json(data))
    board_size = int(data["n"])
    start = tuple(data["start"])
    goal_edge = data["goal_edge"]
    mode = data["mode"]

    print(search_path(successor_state, board_size, start, goal_edge, mode))

    

def state_from_json(data):

    state = []
    board_size = int(data["n"])
    for r in range(board_size):
        board_row = []
        state.append(board_row)
        for q in range(board_size):
            board_row.append(Tile.EMPTY)
    
    for tile in data["board"]:
        state[tile[1]][tile[2]] = tile[0]

