"""
COMP30024 Artificial Intelligence, Semester 1, 2022
Project Part B:

This script contains the allows testing of dijkstra's implementation used
for evaluating cachex positions and checking if the game has been won.

Called from root folder as follows:
python3 -m dijkstra <path to test.json>

JSON example schema:

{
    "n": 5,
    "board": [
        ["blue", 1, 0],
        ["blue", 1, 1],
        ["blue", 1, 3],
        ["blue", 3, 2]
    ],
    "start": [1, 0],
    "goal_edge": "blue_end",
    "player_colour": "blue",
    "mode": "eval"
}

"""

import sys, json
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(main))))


# If you want to separate your code into separate files, put them
# inside the `search` directory (like this one and `util.py`) and
# then import from them like this:
from dijkstra.pathfinding import *
from tenuOS.player import Player
from util.enums import *
from util.general import print_state
from dijkstra.util import *

def main():
    """
    Main program for testing dijkstra on test board state, using json input file
    """

    # convert json input to a dict
    try:
        with open(sys.argv[1]) as file:
            data = json.load(file)
    except IndexError:
        print("usage: python3 -m search path/to/input.json", file=sys.stderr)
        sys.exit(1)

    # store variables from dict as correct data type, converting to enums if needed
    state = state_from_json(data)
    player_colour = ENUM_CONVERSIONS[data["player_colour"]]
    board_size = int(data["n"])
    start = tuple(data["start"])
    goal_edge = ENUM_CONVERSIONS[data["goal_edge"]]
    mode = ENUM_CONVERSIONS[data["mode"]]

    # create dict of occupied cells in format:  (r, q): "colour"
    board_dict = {}
    for tile in data["board"]:
        coord = (tile[1], tile[2])
        board_dict[coord] = tile[0]

    # print the board
    print_board(board_size, board_dict, "THE BOARD")

    # print the state
    print_state(state)

    # print initial variable values
    print("board size = " + str(board_size))
    print("player colour = " + str(player_colour))
    print("start coords = " + str(start))
    print("mode = " + str(mode))
    print("goal edge = " + str(goal_edge))

    # print path cost of shortest path sing dijkstra
    path_cost = search_path(state, player_colour, board_size, start, goal_edge, mode)
    print("\npath cost = " + str(path_cost) + "\n")

def state_from_json(data):
    """
    Takes the dictionary form of the input json and returns a 
    2D array representing the current board state 
    """
    state = []
    board_size = int(data["n"])
    for r in range(board_size):
        board_row = []
        state.append(board_row)
        for q in range(board_size):
            board_row.append(util.constants.EMPTY)
    
    for tile in data["board"]:
        state[tile[1]][tile[2]] = ENUM_CONVERSIONS[tile[0]]

    return state

