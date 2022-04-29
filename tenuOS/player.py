from enum import Enum

class Tile(Enum):
    BLUE = 0
    RED = 1
    EMPTY = 2

class Player:

    def __init__(self, player, n):
        """
        Called once at the beginning of a game to initialise this player.
        Set up an internal representation of the game state.

        The parameter player is the string "red" if your player will
        play as Red, or the string "blue" if your player will play
        as Blue.
        """
        # put your code here
        self.colour = player
        self.board_state = []
        for r in range(n):
            board_row = []
            self.board_state.append(board_row)
            for q in range(n):
                board_row.append(Tile.EMPTY)

    def action(self):
        """
        Called at the beginning of your turn. Based on the current state
        of the game, select an action to play.
        """
        # put your code here
        alpha = float('-inf')
        beta = float('inf')
    
    def turn(self, player, action):
        """
        Called at the end of each player's turn to inform this player of 
        their chosen action. Update your internal representation of the 
        game state based on this. The parameter action is the chosen 
        action itself. 
        
        Note: At the end of your player's turn, the action parameter is
        the same as what your player returned from the action method
        above. However, the referee has validated it at this point.
        """
        # put your code here
        if action[0] == "PLACE":
            r = action[1]
            q = action[2]
            self.board_state[r][q] = player
        # Steal doesn't have r and q, need to store previous move or something, will deal with it in action.


def max_value(state, game, alpha, beta):
    if cutoff_test(state):
        return eval_func(state)

    # add successor_states later, TBD
    for successor_state in successor_states(state):
        alpha = max(alpha, min_value(successor_state, game, alpha, beta))
        if alpha >= beta:
            return beta

    return alpha


def min_value(state, game, alpha, beta):
    if cutoff_test(state):
        return eval_func(state)

    # add successor_states later, TBD
    for successor_state in successor_states(state):
        beta = min(beta, max_value(successor_state, game, alpha, beta))
        if beta <= alpha:
            return alpha

    return beta


def cutoff_test(state):
    # cutoff_depth or goal_state
    return False


def eval_func(state):
    # evaluation function goes here

    # A* eval
    return 0


def successor_states(state):
    successor_states = []
    return successor_states
