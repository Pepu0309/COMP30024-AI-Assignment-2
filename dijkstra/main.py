"""
COMP30024 Artificial Intelligence, Semester 1, 2022
Project Part A: Searching

This script contains the entry point to the program (the code in
`__main__.py` calls `main()`). Your solution starts here!
"""

import sys, json
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(main))))


# If you want to separate your code into separate files, put them
# inside the `search` directory (like this one and `util.py`) and
# then import from them like this:
from dijkstra.pathfinding.pathfinding import *
from tenuOS.player import Player
from util.enums import *
from dijkstra.util import *

debug = False

enum_conversions = {
    "eval": Mode.WIN_DIST,
    "win_test": Mode.WIN_TEST,
    "blue": Tile.BLUE,
    "red": Tile.RED,
    "empty": Tile.EMPTY,
    "blue_start": GoalEdge.BLUE_START,
    "blue_end": GoalEdge.BLUE_END,
    "red_start": GoalEdge.RED_START,
    "red_end": GoalEdge.RED_END,
}

def main():

    try:
        with open(sys.argv[1]) as file:
            data = json.load(file)
    except IndexError:
        print("usage: python3 -m search path/to/input.json", file=sys.stderr)
        sys.exit(1)

    state = state_from_json(data)
    player_colour = enum_conversions[data["player_colour"]]
    board_size = int(data["n"])
    start = tuple(data["start"])
    goal_edge = enum_conversions[data["goal_edge"]]
    mode = enum_conversions[data["mode"]]

    board_dict = {}
    for tile in data["board"]:
        coord = (tile[1], tile[2])
        board_dict[coord] = tile[0]

    print_board(board_size, board_dict, "THE BOARD")

    print_state(state)

    print("board size = " + str(board_size))
    print("player colour = " + str(player_colour))
    print("start coords = " + str(start))
    print("mode = " + str(mode))
    print("goal edge = " + str(goal_edge))

    path_cost = search_path(state, player_colour, board_size, start, goal_edge, mode)
    print("\npath cost = " + str(path_cost) + "\n")

def state_from_json(data):

    state = []
    board_size = int(data["n"])
    for r in range(board_size):
        board_row = []
        state.append(board_row)
        for q in range(board_size):
            board_row.append(Tile.EMPTY)
    
    for tile in data["board"]:
        state[tile[1]][tile[2]] = enum_conversions[tile[0]]

    return state

def print_state(state):

    print("\nstate as a grid: \n")
    for r in reversed(state):
        for q in r:
            print(str(q), end = ", ")
        print("")
    print("")