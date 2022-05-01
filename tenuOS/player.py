from cmath import inf
from util.enums import *
from dijkstra.pathfinding.pathfinding import search_path

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
        self.board_size = n
        for r in range(self.board_size):
            board_row = []
            self.board_state.append(board_row)
            for q in range(self.board_size):
                board_row.append(Tile.EMPTY)

    def action(self):
        """
        Called at the beginning of your turn. Based on the current state
        of the game, select an action to play.
        """
        # put your code here
        alpha = -(float('inf'))
        beta = float('inf')

        best_move = None
        for potential_move in get_successor_states(self.board_state, self.board_size, self.player_colour):
            # For each move their, evaluation function value should be the minimum value of its successor states
            # due to game theory (opponent plays the lowest value move). Hence, we call min_value for all
            # the potential moves available to us in this current turn.
            cur_move_value = min_value(potential_move, self.board_size, alpha, beta, 1, self.player_colour)

            if cur_move_value > alpha:
                alpha = cur_move_value
                best_move = potential_move

        print(best_move)
    
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


# Pseudocode from lectures but with "game" variable omitted (though probably included through board_size and
# player_colour
def max_value(state, board_size, alpha, beta, depth, player_colour):
    if cutoff_test(state, depth):
        return eval_func(state)

    successor_states = get_successor_states(state, board_size, player_colour)
    for successor_state in successor_states:
        alpha = max(alpha, min_value(successor_state, board_size, alpha, beta, depth+1, player_colour))
        if alpha >= beta:
            return beta

    return alpha


def min_value(state, board_size, alpha, beta, depth, player_colour):
    if cutoff_test(state, depth):
        return eval_func(state)

    successor_states = get_successor_states(state, board_size, player_colour)
    for successor_state in successor_states:
        beta = min(beta, max_value(successor_state, board_size, alpha, beta, depth+1, player_colour))
        if beta <= alpha:
            return alpha

    return beta


def cutoff_test(state, depth):
    # cutoff_depth or terminal_state
    if depth == 4:
        return True

    return False


def eval_func(self, state):
    
    def win_distance_difference(state, board_size, player_colour):

        def starting_edge_red(state):

            # record the frequencies of the different tile colours/types on both the starting
            # and ending goal edges for red
            r = 0
            tile_dict_start = {Tile.RED: 0, Tile.BLUE: 0, Tile.EMPTY: 0}
            for q in range(board_size):
                tile_dict_start[state[r][q]] += 1

            r = board_size - 1
            tile_dict_end = {Tile.RED: 0, Tile.BLUE: 0, Tile.EMPTY: 0}
            for q in range(board_size):
                tile_dict_end[state[r][q]] += 1

            # return the edge with the fewest blue tiles, if a tie occurs
            # return the edge with the most red tiles
            if tile_dict_start[Tile.BLUE] > tile_dict_end[Tile.BLUE]:
                return GoalEdge.RED_END
            elif tile_dict_end[Tile.BLUE] - tile_dict_start[Tile.BLUE]:
                return GoalEdge.RED_START
            else:
                if tile_dict_start[Tile.RED] > tile_dict_end[Tile.RED]:
                    return GoalEdge.RED_START
                else:
                    return GoalEdge.RED_END

        def starting_edge_blue(state):

            # record the frequencies of the different tile colours/types on both the starting
            # and ending goal edges for blue
            q = 0
            tile_dict_start = {Tile.RED: 0, Tile.BLUE: 0, Tile.EMPTY: 0}
            for r in range(board_size):
                tile_dict_start[state[r][q]] += 1

            q = board_size - 1
            tile_dict_end = {Tile.RED: 0, Tile.BLUE: 0, Tile.EMPTY: 0}
            for r in range(board_size):
                tile_dict_end[state[r][q]] += 1

            # return the edge with the fewest red tiles, if a tie occurs
            # return the edge with the most rbluetiles
            if tile_dict_start[Tile.RED] > tile_dict_end[Tile.RED]:
                return GoalEdge.BLUE_END
            elif tile_dict_end[Tile.RED] - tile_dict_start[Tile.RED]:
                return GoalEdge.BLUE_START
            else:
                if tile_dict_start[Tile.BLUE] > tile_dict_end[Tile.BLUE]:
                    return GoalEdge.BLUE_START
                else:
                    return GoalEdge.BLUE_END

        # if starting at start edge have init start node (0, 0)
        # if starting at end edge have init start node (0, board_size - 1)
        starting_edge = starting_edge_blue(state)
        q = 0 if starting_edge == GoalEdge.BLUE_START else board_size - 1
        goal_edge = GoalEdge.BLUE_END if starting_edge == GoalEdge.BLUE_START else GoalEdge.BLUE_START
        min_win_dist_blue = inf
        for r in range(board_size):
            if state[r][q] == Tile.RED: pass
            temp_path_cost = search_path(state, Tile.BLUE, board_size, (r, q), goal_edge, Mode.WIN_DIST)
            if temp_path_cost < min_win_dist_blue:
                min_win_dist_blue = temp_path_cost
        
        # if starting at start edge have init start node (0, 0)
        # if starting at end edge have init start node (0, board_size - 1)
        starting_edge = starting_edge_red(state)
        r = 0 if starting_edge == GoalEdge.RED_START else board_size - 1
        goal_edge = GoalEdge.RED_END if starting_edge == GoalEdge.RED_START else GoalEdge.RED_START
        min_win_dist_red = inf
        for q in range(board_size):
            if state[r][q] == Tile.RED: pass
            temp_path_cost = search_path(state, Tile.BLUE, board_size, (r, q), goal_edge, Mode.WIN_DIST)
            if temp_path_cost < min_win_dist_red:
                min_win_dist_red = temp_path_cost

        # return win distance difference value such that higher is better for our colour
        win_dist_diff = min_win_dist_red - min_win_dist_blue
        return win_dist_diff if player_colour == Tile.BLUE else -win_dist_diff

    win_dist_diff = win_distance_difference(state, self.board_size, self.player_colour)
    evaluation = 1 * win_dist_diff
    return evaluation




def get_successor_states(state, board_size, player_colour):
    # Create successor for the moves using the player colour
    successor_states = []

    for r in range(board_size):
        for q in range(board_size):
            successor_states.append(SuccessorState(state[:], (r, q), player_colour))

    return successor_states

class SuccessorState:
    def __init__(self, state, move, player_colour):
        self.move_r = move[0]
        self.move_q = move[1]
        self.player_colour = player_colour
        state[self.move_r][self.move_q] = player_colour
        self.state = state

    def capture(self):
        r = self.move_r
        q = self.move_q

        # Temporary placeholder, implement logic to determine opponent colour later
        opponent_colour = "red"

        # If there is an occupied cell of the same colour to the bottom of this current move
        if self.state[r-2][q+1] == self.player_colour:
            # Then, if there's also occupied cells belonging to the opponent to the bottom right and bottom left of
            # this move, then this move is a capture.
            if self.state[r-1][q] == opponent_colour and self.state[r-1][q+1] == opponent_colour:
                return True
        # If there is an occupied cell of the same colour above this current move
        elif self.state[r+2][q-1] == self.player_colour:
            # Then, if there's also occupied cells belonging to the opponent to the top right and top left of
            # this move, this move is a capture.
            if self.state[r+1][q-1] == opponent_colour and self.state[r+1][q] == opponent_colour:
                return True
        # If there is an occupied cell of the same colour to the right of this current move
        elif self.state[r][q-1] == self.player_colour:
            # Then, if there's also occupied cells belonging to the opponent to the top right and bottom right of
            # this move, this move is a capture.
            if self.state[r+1][q-1] == opponent_colour and self.state[r-1][q] == opponent_colour:
                return True
        # If there is an occupied cell of the same colour to the left of this current move
        elif self.state[r][q+1] == self.player_colour:
            # Then, if there's also occupied cells belonging to the opponent to the top left and bottom left of
            # this move, this move is a capture.
            if self.state[r+1][q] == opponent_colour and self.state[r-1][q+1] == opponent_colour:
                return True


