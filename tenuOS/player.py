from cmath import inf
from util.enums import *
from dijkstra.pathfinding import search_path
from util.general import *
import copy
import util.constants
import numpy as np
import gc

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
        if player == "red":
            self.player_colour = util.constants.RED
        elif player == "blue":
            self.player_colour = util.constants.BLUE

        self.board_size = n
        # self.board_state = []
        # for r in range(self.board_size):
        #     board_row = []
        #     self.board_state.append(board_row)
        #     for q in range(self.board_size):
        #         board_row.append(util.constants.EMPTY)
        self.board_state = np.full((n, n), util.constants.EMPTY, dtype="int8")
        self.current_turn = 0
        self.my_last_move = None
        self.opponent_last_move = None
        self.steal_coords = None

    def action(self):
        """
        Called at the beginning of your turn. Based on the current state
        of the game, select an action to play.
        """
        # put your code here
        alpha = -(float('inf'))
        beta = float('inf')

        # print("state at start of action()")
        # print_state(self.board_state)
        move = None

        # -------------------------------------------Opening Playbook--------------------------------------------------
        if self.current_turn == 0:
            move = ("PLACE", 1, 1)
        elif self.current_turn == 1:
            r = self.opponent_last_move[1]
            q = self.opponent_last_move[2]

            largest_board_index = self.board_size - 1

            if self.board_size % 4 <= 2:
                divider = self.board_size // 4
            else:
                divider = self.board_size // 4 + 1

            if is_connected_diagonal(r, q, self.board_size) and (abs(r-q) <= (self.board_size - divider)):
                move = ("STEAL", )
            elif divider <= r <= (largest_board_index - divider) and \
                    divider <= q <= (largest_board_index - divider):
                move = ("STEAL",)
            else:
                move = ("PLACE", self.board_size // 2, self.board_size // 2)
        else:
            best_move = None
            opponent_last = (self.steal_coords[0], self.steal_coords[1]) if self.opponent_last_move[0] == "STEAL" else (self.opponent_last_move[1], self.opponent_last_move[2])
            my_last = (self.steal_coords[0], self.steal_coords[1]) if self.my_last_move[0] == "STEAL" else (self.my_last_move[1], self.my_last_move[2])
            #print(my_last)
            #print(opponent_last)
            for successor_state in self.get_successor_states(self.board_state, self.board_size, self.player_colour, opponent_last, my_last):
                # For each move their, evaluation function value should be the minimum value of its successor states
                # due to game theory (opponent plays the lowest value move). Hence, we call min_value for all
                # the potential moves available to us in this current turn.
                cur_move_value = self.min_value(successor_state, self.board_size, alpha, beta, 1, (self.player_colour + 1) % 2)
                gc.collect()

                if cur_move_value > alpha:
                    alpha = cur_move_value
                    best_move = successor_state
                    #print("hi")

            # print(best_move)

            move = ("PLACE", best_move.move[0], best_move.move[1])

        self.my_last_move = move

        return move
    
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
        
        # store player colour using our own defined constants
        if player == "red":
            player_colour = util.constants.RED
        elif player == "blue":
            player_colour = util.constants.BLUE

        # store axial coords of the action if the action was to place
        if action[0] == "PLACE":
            r = action[1]
            q = action[2]
        # if the action was steal, retrieve coords of reflected tile from
        # knowledge of first played move
        elif action[0] == "STEAL":
            if player_colour == self.player_colour:
                q = self.opponent_last_move[1]
                r = self.opponent_last_move[2]
            else:
                q = self.my_last_move[1]
                r = self.my_last_move[2]
            # remove the tile placed by the player whose move was stolen
            self.board_state[q][r] = util.constants.EMPTY
            # store the coords where the reflected tile will be placed
            self.steal_coords = (r, q)

        # place the played move in board state variable
        self.board_state[r][q] = player_colour

        # update knowledge of last played move for us or opponent
        if player_colour == self.player_colour:
            self.my_last_move = action
        else:
            self.opponent_last_move = action

        # update counter of moves played
        self.current_turn += 1

    def cutoff_test(self, state, depth):
        """
        Test a hypothetical state at a given depth and determine if either
        the terminal state has been reached or the depth limit has been reached.
        """
        
        # cutoff_depth or terminal_state
        if depth == 4:
            return True

        return False

    def eval_func(self, state):
        """
        Evaluates a state, with a higher number meaning more payoff for our player.
        """

        # calculate all features
        win_dist_diff = self.win_distance_difference(state, self.board_size, self.player_colour)
        tile_difference = self.tile_difference(state)
        two_bridge_count_diff = self.two_bridge_count_diff(state)

        # sum features according to weights
        evaluation = 0.4 * win_dist_diff + 0.5 * tile_difference + 0.1 * two_bridge_count_diff

        return evaluation

    def win_distance_difference(self, state, board_size, player_colour):

        # record the frequencies of the different tile colours/types on both the starting
        # and ending goal edges for red
        r = 0
        tile_dict_start = {util.constants.RED: 0, util.constants.BLUE: 0, util.constants.EMPTY: 0}
        for q in range(board_size):
            tile_dict_start[state[r][q]] += 1

        r = board_size - 1
        tile_dict_end = {util.constants.RED: 0, util.constants.BLUE: 0, util.constants.EMPTY: 0}
        for q in range(board_size):
            tile_dict_end[state[r][q]] += 1

        # return the edge with the fewest blue tiles, if a tie occurs
        # return the edge with the most red tiles
        if tile_dict_start[util.constants.BLUE] > tile_dict_end[util.constants.BLUE]:
            starting_edge_red = BoardEdge.RED_END
        elif tile_dict_end[util.constants.BLUE] - tile_dict_start[util.constants.BLUE]:
            starting_edge_red = BoardEdge.RED_START
        else:
            if tile_dict_start[util.constants.RED] > tile_dict_end[util.constants.RED]:
                starting_edge_red = BoardEdge.RED_START
            else:
                starting_edge_red = BoardEdge.RED_END

        # record the frequencies of the different tile colours/types on both the starting
        # and ending goal edges for blue
        q = 0
        tile_dict_start = {util.constants.RED: 0, util.constants.BLUE: 0, util.constants.EMPTY: 0}
        for r in range(board_size):
            tile_dict_start[state[r][q]] += 1

        q = board_size - 1
        tile_dict_end = {util.constants.RED: 0, util.constants.BLUE: 0, util.constants.EMPTY: 0}
        for r in range(board_size):
            tile_dict_end[state[r][q]] += 1

        # return the edge with the fewest red tiles, if a tie occurs
        # return the edge with the most rbluetiles
        if tile_dict_start[util.constants.RED] > tile_dict_end[util.constants.RED]:
            starting_edge_blue = BoardEdge.BLUE_END
        elif tile_dict_end[util.constants.RED] - tile_dict_start[util.constants.RED]:
            starting_edge_blue = BoardEdge.BLUE_START
        else:
            if tile_dict_start[util.constants.BLUE] > tile_dict_end[util.constants.BLUE]:
                starting_edge_blue = BoardEdge.BLUE_START
            else:
                starting_edge_blue = BoardEdge.BLUE_END

        # if starting at start edge have init start node (0, 0)
        # if starting at end edge have init start node (0, board_size - 1)
        q = 0 if starting_edge_blue == BoardEdge.BLUE_START else board_size - 1
        goal_edge = BoardEdge.BLUE_END if starting_edge_blue == BoardEdge.BLUE_START else BoardEdge.BLUE_START
        min_win_dist_blue = board_size * 2
        for r in range(board_size):
            if state[r][q] == util.constants.RED: pass
            temp_path_cost = search_path(state, util.constants.BLUE, board_size, (r, q), goal_edge, Mode.WIN_DIST)
            if temp_path_cost and temp_path_cost < min_win_dist_blue:
                min_win_dist_blue = temp_path_cost
        
        # if starting at start edge have init start node (0, 0)
        # if starting at end edge have init start node (0, board_size - 1)
        r = 0 if starting_edge_red == BoardEdge.RED_START else board_size - 1
        goal_edge = BoardEdge.RED_END if starting_edge_red == BoardEdge.RED_START else BoardEdge.RED_START
        min_win_dist_red = board_size * 2
        for q in range(board_size):
            if state[r][q] == util.constants.BLUE: pass
            temp_path_cost = search_path(state, util.constants.RED, board_size, (r, q), goal_edge, Mode.WIN_DIST)
            if temp_path_cost and temp_path_cost < min_win_dist_red:
                min_win_dist_red = temp_path_cost

        #print(min_win_dist_blue)
        #print(min_win_dist_red)
        # return win distance difference value such that higher is better for our colour
        win_dist_diff = min_win_dist_red - min_win_dist_blue
        return win_dist_diff if player_colour == util.constants.BLUE else -win_dist_diff

    def tile_difference(self, state):
        """
        Takes a state and returns the difference in the number of tiles that exist on the
        board placed by our player and our opponent. 
        
        A larger value means more of our colour tiles than the opponents colour.
        """

        player_tile_count = 0
        opponent_tile_count = 0

        # loop through all tiles on the board and record the number of tiles
        # which exist on the board placed by our own player and the opponent
        for r in range(self.board_size):
            for q in range(self.board_size):
                if state[r][q] == self.player_colour:
                    player_tile_count += 1
                elif state[r][q] == (self.player_colour + 1) % 2:
                    opponent_tile_count += 1

        # calculate the difference, with a large value meaning our player
        # has more tiles on the board
        tile_difference = player_tile_count - opponent_tile_count
        return tile_difference

    def two_bridge_count_diff(self, state):
        """
        to be documented
        """

        player_two_bridge_count = 0
        opponent_two_bridge_count = 0

        for r in range(self.board_size):
            for q in range(self.board_size):
                if state[r][q] != util.constants.EMPTY:
                    cur_cell_occupied_colour = state[r][q]
                    if cur_cell_occupied_colour == self.player_colour:
                        player_two_bridge_count += two_bridge_check("count", state, r, q,
                                                                    self.board_size, cur_cell_occupied_colour)
                    elif cur_cell_occupied_colour == (self.player_colour + 1) % 2:
                        opponent_two_bridge_count += two_bridge_check("count", state, r, q,
                                                                    self.board_size, cur_cell_occupied_colour)

        return player_two_bridge_count - opponent_two_bridge_count

    # Pseudocode from lectures but with "game" variable omitted (though probably included through board_size and
    # player_colour
    def max_value(self, input_state, board_size, alpha, beta, depth, player_colour):
        """
        Max value function is called on states resulting from a move of the opponent's colour,
        calls mix value function on states resulting from candidate moves by our player.
        """

        if self.cutoff_test(input_state.state, depth):
            return self.eval_func(input_state.state)

        successor_states = self.get_successor_states(input_state.state, board_size, player_colour, input_state.move, input_state.prev_move)
        for successor_state in successor_states:
            alpha = max(alpha, self.min_value(successor_state, board_size, alpha, beta, depth + 1, (player_colour + 1) % 2))
            if alpha >= beta:
                return beta

        return alpha

    def min_value(self, input_state, board_size, alpha, beta, depth, player_colour):
        """
        Min value function is called on states resulting from a move of our own colour,
        calls max value function on states resulting from candidate moves by the opponent.
        """

        if self.cutoff_test(input_state.state, depth):
            return self.eval_func(input_state.state)

        successor_states = self.get_successor_states(input_state.state, board_size, player_colour, input_state.move, input_state.prev_move)
        for successor_state in successor_states:
            beta = min(beta, self.max_value(successor_state, board_size, alpha, beta, depth + 1, (player_colour + 1) % 2))
            if beta <= alpha:
                return alpha

        return beta

    def get_successor_states(self, state, board_size, player_colour, last_move, prior_move):
        """
        Takes a state as input, the player colour of whose turn it is in the given state,
        and returns a list of possible 'successor states' resulting from different moves
        that could be played. 
        
        Generates successor states in rings about the tiles placed
        in the most recent two moves to optimise the order in which alpha-beta considers
        moves at depth 0, as the best move is likely to be near these tiles.
        """

        # list for successor states, dict to store which states have already been added
        successor_states = []
        added = {}

        # initialise dictionary to false for all tiles
        for r in range(board_size):
            for q in range(board_size):
                added[(r, q)] = False
   
        # dictionary constant used to iterate in spiral/circles about a central tile
        NEXT = {util.constants._30_DEG:  (-1,  1),
                util.constants._90_DEG:  (-1,  0),
                util.constants._150_DEG: (0 , -1),
                util.constants._210_DEG: (1 , -1),
                util.constants._270_DEG: (1 ,  0),
                util.constants._330_DEG: (0 ,  1)}
        LAYERS = 2 # num layers to explore about a tile

        # repeat state generation about most recent move for both players
        for move in (last_move, prior_move):

            # arbitrary start at 30 degrees north 
            angle = util.constants._30_DEG
            # this equates to +1 in the r axis
            r, q = move[0] + 1, move[1]
            
            # search all tiles LAYERS about tile placed by move
            for layer in range(LAYERS):
                # repeat for each side of the layer (6 times as hexagons)
                for i in range(len(NEXT)):
                    # the number of cells on each edge == the layer
                    for j in range(layer):
                        # if tile is valid (within the board dimensions), not yet had its successor state added,
                        # and is currently empty, generate successor state and add to list
                        if is_valid_cell(r, q, board_size) and (not added[(r, q)]) and state[r][q] == util.constants.EMPTY:
                            # update added dict
                            added[(r, q)] = True
                            successor_states.append(SuccessorState(state, (r, q), last_move, player_colour, board_size))
                        # adjust r and q according the current angle 
                        r += NEXT[angle][0]
                        q += NEXT[angle][1]
                    # move to next edge
                    angle = (angle + 1) % len(NEXT)

        return successor_states

class SuccessorState:

    def __init__(self, state, move, prev_move, player_colour, board_size):

        # generate copy of the state
        self.state = np.copy(state)
        # store the move being considered
        self.move = move
        # store the most recent move in the state parameter
        self.prev_move = prev_move
        # store the player colour who is playing the move being considered
        self.player_colour = player_colour
        self.board_size = board_size

        # apply the considered move to state instance variable
        self.apply_move()

    def apply_move(self):

        # Change the cell to player's colour
        self.state[self.move[0]][self.move[1]] = self.player_colour

        r = self.move[0]
        q = self.move[1]
        board_size = self.board_size

        cells_to_remove = []
        opponent_colour = (self.player_colour + 1) % 2

        # Check for possible captures and tag the cells to be removed from captures
        # ----------------------------------2-Bridge Capture Cases-------------------------------------------
        two_bridge_check("capture", self.state, r, q, board_size, self.player_colour, cells_to_remove)
        # -------------------------------------All Cells Adjacent Cases--------------------------------------------
        # If there is an occupied cell of the same colour distance 1 to the left of this current move
        if is_valid_cell(r, q-1, board_size):
            if self.state[r][q-1] == self.player_colour:
                # Then, if there's also occupied cells belonging to the opponent distance 1 to the top left and
                # bottom left of this move, this move is a capture.
                if is_valid_cell(r+1, q-1, board_size) and is_valid_cell(r-1, q, board_size):
                    if self.state[r+1][q-1] == opponent_colour and self.state[r-1][q] == opponent_colour:
                        cells_to_remove.append((r + 1, q - 1))
                        cells_to_remove.append((r - 1, q))

        # If there is an occupied cell of the same colour distance 1 to the right of this current move
        if is_valid_cell(r, q+1, board_size):
            if self.state[r][q+1] == self.player_colour:
                # Then, if there's also occupied cells belonging to the opponent distance 1 to the top right and
                # bottom right of this move, this move is a capture.
                if is_valid_cell(r+1, q, board_size) and is_valid_cell(r-1, q+1, board_size):
                    if self.state[r+1][q] == opponent_colour and self.state[r-1][q+1] == opponent_colour:
                        cells_to_remove.append((r + 1, q))
                        cells_to_remove.append((r - 1, q + 1))

        # If there is an occupied cell of the same colour distance 1 to the top left of this current move
        if is_valid_cell(r+1, q-1, board_size):
            if self.state[r+1][q-1] == self.player_colour:
                # Then, if there's also occupied cells belonging to the opponent distance 1 to the left and top right
                # of this move, this move is a capture.
                if is_valid_cell(r, q-1, board_size) and is_valid_cell(r+1, q, board_size):
                    if self.state[r][q-1] == opponent_colour and self.state[r+1][q] == opponent_colour:
                        cells_to_remove.append((r, q - 1))
                        cells_to_remove.append((r + 1, q))

        # If there is an occupied cell of the same colour distance 1 to the bottom right of this current move
        if is_valid_cell(r-1, q+1, board_size):
            if self.state[r-1][q+1] == self.player_colour:
                # Then, if there's also occupied cells belonging to the opponent distance 1 to the bottom left and right
                # of this move, this move is a capture.
                if is_valid_cell(r-1, q, board_size) and is_valid_cell(r, q+1, board_size):
                    if self.state[r-1][q] == opponent_colour and self.state[r][q+1] == opponent_colour:
                        cells_to_remove.append((r - 1, q))
                        cells_to_remove.append((r, q+1))

        # If there is an occupied cell of the same colour distance 1 to the bottom left of this current move
        if is_valid_cell(r-1, q, board_size):
            if self.state[r-1][q] == self.player_colour:
                # Then, if there's also occupied cells belonging to the opponent distance 1 to the left and bottom right
                # of this move, this move is a capture.
                if is_valid_cell(r, q-1, board_size) and is_valid_cell(r-1, q+1, board_size):
                    if self.state[r][q-1] == opponent_colour and self.state[r-1][q+1] == opponent_colour:
                        cells_to_remove.append((r, q - 1))
                        cells_to_remove.append((r - 1, q + 1))

        # If there is an occupied cell of the same colour distance 1 to the top right of this current move
        if is_valid_cell(r+1, q, board_size):
            if self.state[r+1][q] == self.player_colour:
                # Then, if there's also occupied cells belonging to the opponent distance 1 to the top left and right
                # of this move, this move is a capture.
                if is_valid_cell(r+1, q-1, board_size) and is_valid_cell(r, q+1, board_size):
                    if self.state[r+1][q-1] == opponent_colour and self.state[r][q+1] == opponent_colour:
                        cells_to_remove.append((r + 1, q - 1))
                        cells_to_remove.append((r, q + 1))

        for cell in cells_to_remove:
            self.state[cell[0]][cell[1]] = util.constants.EMPTY


"""
mode: A parameters which functions like a flag. Either "capture" or "count" should be passed.
state: A parameter which is the board state to evaluate. Either a 2D numpy array (what we use) or a 2D Python list.
r: r-coordinate of the move to be evaluated in a capture or two-bridge count.
q: q-coordinate of the move to be evaluated in a capture or two-bridge count.
board_size: The size of the board the Cachex game is being played on.
player_colour: The player_colour of the move to check.
cells_to_remove: Optional parameter; used with "capture". A list of the moves to be removed that is passed in by the
                 calling function and returned to them.
"""
def two_bridge_check(mode, state, r, q, board_size, player_colour, cells_to_remove = None):


    # If we're trying to check for and apply captures in a 2-bridge, then we're checking for opponent_colour
    # in the 2 adjacent cells in between the 2-bridge. If we're trying to count the number of 2 bridges, then we want
    # the 2 adjacent cells in between the 2-bridge to be empty.
    if mode == "capture":
        tile_status = (player_colour + 1) % 2
    elif mode == "count":
        tile_status = util.constants.EMPTY

    two_bridge_count = 0
    # For counting, we only need to check the 2-bridges to the top. Any 2-bridges towards the bottom will be picked up
    # by a later check from the bottom to the top. This avoids counting duplicates. For capture-checking, we need to
    # check in all 6 directions.

    # If there is an occupied cell of the same colour distance 2 away above this current move
    if is_valid_cell(r + 2, q - 1, board_size):
        if state[r + 2][q - 1] == player_colour:
            # Then, if there's also occupied cells belonging to the opponent distance 1 to the top left and
            # top right of this move, this move is a capture.
            if is_valid_cell(r + 1, q - 1, board_size) and is_valid_cell(r + 1, q, board_size):
                if state[r + 1][q - 1] == tile_status and state[r + 1][q] == tile_status:
                    if mode == "capture":
                        cells_to_remove.append((r + 1, q - 1))
                        cells_to_remove.append((r + 1, q))
                    elif mode == "count":
                        two_bridge_count += 1

    # If there is an occupied cell of the same colour distance 2 away to the top-left of this current move.
    if is_valid_cell(r + 1, q - 2, board_size):
        if state[r + 1][q - 2] == player_colour:
            # Then, if there's also occupied cells belonging to the opponent distance 1 to the top left and left
            # of this move, this move is a capture.
            if is_valid_cell(r + 1, q - 1, board_size) and is_valid_cell(r, q - 1, board_size):
                if state[r + 1][q - 1] == tile_status and state[r][q - 1] == tile_status:
                    if mode == "capture":
                        cells_to_remove.append((r + 1, q - 1))
                        cells_to_remove.append((r, q - 1))
                    elif mode == "count":
                        two_bridge_count += 1

    # If there is an occupied cell of the same colour distance 2 away to the top right of this current move
    if is_valid_cell(r + 1, q + 1, board_size):
        if state[r + 1][q + 1] == player_colour:
            # Then, if there's also occupied cells belonging to the opponent distance 1 to the right and top right
            # of this move, then this move is a capture.
            if is_valid_cell(r, q + 1, board_size) and is_valid_cell(r + 1, q, board_size):
                if state[r][q + 1] == tile_status and state[r + 1][q] == tile_status:
                    if mode == "capture":
                        cells_to_remove.append((r, q + 1))
                        cells_to_remove.append((r + 1, q))
                    elif mode == "count":
                        two_bridge_count += 1

    # Only check the bottom cases when checking for capture
    if mode == "capture":
        # If there is an occupied cell of the same colour distance 2 away to the bottom-left of this current move.
        if is_valid_cell(r - 1, q - 1, board_size):
            if state[r - 1][q - 1] == player_colour:
                # Then, if there's also occupied cells belonging to the opponent distance 1 to the left and bottom left
                # of this move, this move is a capture.
                if is_valid_cell(r, q - 1, board_size) and is_valid_cell(r - 1, q, board_size):
                    if state[r][q - 1] == tile_status and state[r - 1][q] == tile_status:
                        if mode == "capture":
                            cells_to_remove.append((r, q - 1))
                            cells_to_remove.append((r - 1, q))
                        elif mode == "count":
                            two_bridge_count += 1

        # If there is an occupied cell of the same colour distance 2 away to the bottom of this current move
        if is_valid_cell(r - 2, q + 1, board_size):
            if state[r - 2][q + 1] == player_colour:
                # Then, if there's also occupied cells belonging to the opponent distance 1 to the bottom left and
                # bottom right of this move, then this move is a capture.
                if is_valid_cell(r - 1, q, board_size) and is_valid_cell(r - 1, q + 1, board_size):
                    if state[r - 1][q] == tile_status and state[r - 1][q + 1] == tile_status:
                        if mode == "capture":
                            cells_to_remove.append((r - 1, q))
                            cells_to_remove.append((r - 1, q + 1))
                        elif mode == "count":
                            two_bridge_count += 1

        # If there is an occupied cell of the same colour distance 2 away to the bottom right of this current move
        if is_valid_cell(r - 1, q + 2, board_size):
            if state[r - 1][q + 2] == player_colour:
                # Then, if there's also occupied cells belonging to the opponent distance 1 to the bottom right and
                # right of this move, then this move is a capture.
                if is_valid_cell(r - 1, q + 1, board_size) and is_valid_cell(r, q + 1, board_size):
                    if state[r - 1][q + 1] == tile_status and state[r][q + 1] == tile_status:
                        if mode == "capture":
                            cells_to_remove.append((r - 1, q + 1))
                            cells_to_remove.append((r, q + 1))
                        elif mode == "count":
                            two_bridge_count += 1

    if mode == "capture":
        return cells_to_remove
    elif mode == "count":
        return two_bridge_count
