from enum import Enum

class Tile(Enum):
    BLUE = 0
    RED = 1
    EMPTY = 2

class Mode(Enum):
    EVAL = 0
    WIN_TEST = 1

class GoalEdge(Enum):
    BLUE_START = 0
    BLUE_END = 1
    RED_START = 2
    RED_END = 3