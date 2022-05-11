from enum import Enum
import tenuOS.util.constants

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
    "win_dist": Mode.WIN_DIST,
    "win_test": Mode.WIN_TEST,
    "blue": tenuOS.util.constants.BLUE,
    "red": tenuOS.util.constants.RED,
    "empty": tenuOS.util.constants.EMPTY,
    "blue_start": BoardEdge.BLUE_START,
    "blue_end": BoardEdge.BLUE_END,
    "red_start": BoardEdge.RED_START,
    "red_end": BoardEdge.RED_END,
}
