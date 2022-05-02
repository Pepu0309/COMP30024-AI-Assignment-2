from enum import Enum

class Tile(Enum):
    BLUE = 0
    RED = 1
    EMPTY = 2

class Mode(Enum):
    WIN_DIST = 0
    WIN_TEST = 1

class GoalEdge(Enum):
    BLUE_START = 0
    BLUE_END = 1
    RED_START = 2
    RED_END = 3

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
