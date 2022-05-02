from enum import Enum

class Tile(Enum):
    BLUE = 0
    RED = 1
    EMPTY = 2

class Mode(Enum):
    WIN_DIST = 0
    WIN_TEST = 1

class BoardEdge(Enum):
    BLUE_START = 0
    BLUE_END = 1
    RED_START = 2
    RED_END = 3

ENUM_CONVERSIONS = {
    "eval": Mode.WIN_DIST,
    "win_test": Mode.WIN_TEST,
    "blue": Tile.BLUE,
    "red": Tile.RED,
    "empty": Tile.EMPTY,
    "blue_start": BoardEdge.BLUE_START,
    "blue_end": BoardEdge.BLUE_END,
    "red_start": BoardEdge.RED_START,
    "red_end": BoardEdge.RED_END,
}
