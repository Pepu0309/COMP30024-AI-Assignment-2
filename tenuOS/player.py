from cmath import inf
from tenuOS.dijkstra.pathfinding import search_path
import tenuOS.util.constants
from tenuOS.util.general import *
from tenuOS.util.enums import *
import numpy as np
import gc
import time

class Player:

    def __init__(self, player, n):
        """
        Called once at the beginning of a game to initialise this player.
        Set up an internal representation of the game state.

        The parameter player is the string "red" if your player will
        play as Red, or the string "blue" if your player will play
        as Blue.
        """
        init_start_time = time.process_time()

        if player == "red":
            self.player_colour = tenuOS.util.constants.RED
        elif player == "blue":
            self.player_colour = tenuOS.util.constants.BLUE

        self.board_size = n
        self.board_state = np.full((n, n), tenuOS.util.constants.EMPTY, dtype="int8")

        # Used for reverting to basic strategy in case nearing time limit
        self.time_limit = n ** 2
        self.time_elapsed = 0
        self.last_search_time = 0

        # Used for dynamic depth
        self.depth_limit = 2
        self.TILE_COUNT = n ** 2
        self.max_branching_factor = self.TILE_COUNT

        # storing number of turns for terminal state optimzation
        self.current_turn = 0

        # storing number of tiles for terminal state optimization
        # and endgame detection
        self.num_tiles = 0

        # storing each players last moves for successor state generation
        self.my_last_move = None
        self.opponent_last_move = None

        # store the coordinates of the steal, after reflection
        self.steal_coords = None

        self.time_elapsed += time.process_time() - init_start_time

    def action(self):
        """
        Called at the beginning of your turn. Based on the current state
        of the game, select an action to play.
        """
        action_start_time = time.process_time()
        alpha = -(float('inf'))
        beta = float('inf')

        move = None

        # -------------------------------------------Opening Playbook--------------------------------------------------
        if self.current_turn <= 1:
            if self.board_size % 4 <= 2:
                divider = self.board_size // 4
            else:
                divider = self.board_size // 4 + 1

            largest_board_index = self.board_size - 1

            if self.current_turn == 0:
                # On board size 3, the strong diagonals are just 1 away from the middle and are too strong.
                if self.board_size == 3:
                    move = ("PLACE", 1, 2)
                # Place on the tile that's on the strong diagonal on the bottom right distance 1 outside the
                # parallelogram we established would be good to steal from.
                elif self.board_size >= 4:
                    move = ("PLACE", divider - 1, largest_board_index - divider + 1)
            elif self.current_turn == 1:
                r = self.opponent_last_move[1]
                q = self.opponent_last_move[2]

                if is_connected_diagonal(r, q, self.board_size) and (abs(r-q) <= (self.board_size - divider)):
                    move = ("STEAL", )
                elif divider <= r <= (largest_board_index - divider) and \
                        divider <= q <= (largest_board_index - divider):
                    move = ("STEAL",)
                else:
                    move = ("PLACE", self.board_size // 2, self.board_size // 2)
        else:

            best_move = None
            
            # storing last move of both players as coordinate tuples, ternary to deal with cases where STEAL was played recently
            opponent_last = (self.steal_coords[0], self.steal_coords[1]) if self.opponent_last_move[0] == "STEAL" else (self.opponent_last_move[1], self.opponent_last_move[2])
            my_last = (self.steal_coords[0], self.steal_coords[1]) if self.my_last_move[0] == "STEAL" else (self.my_last_move[1], self.my_last_move[2])

            # retrieve successor states for the current board states
            successor_states = self.get_successor_states(self.board_state, self.board_size, self.num_tiles, self.player_colour, opponent_last, my_last)

            start_search_time = time.process_time()
            # loop over successor states, this acts as depth 0 of alpha-beta search
            for successor_state in successor_states:

                # if we're nearing the time limit, switch back to a greedy strategy and if we are still running out
                # of time, switch to just evaluating the tile difference
                if self.time_elapsed + self.last_search_time >= tenuOS.util.constants.GREEDY_THRESHOLD * self.time_limit:
                    terminal = self.terminal_state_check(successor_state)
                    if self.time_elapsed + self.last_search_time >= tenuOS.util.constants.TILE_DIFF_ONLY_THRESHOLD * self.time_limit:
                        cur_move_value = inf if terminal else self.tile_difference(successor_state.state)
                    else:
                        cur_move_value = inf if terminal else self.eval_func(successor_state.state)
                # for each move, evaluation function value should be the minimum value of its successor states
                # due to game theory (opponent plays the lowest value move) hence, we call min_value for all
                # the potential moves available to us in this current turn
                else:
                    cur_move_value = self.min_value(successor_state, self.board_size, alpha, beta, 1, (self.player_colour + 1) % 2)

                gc.collect()

                # if move is better than previous best, store 
                if cur_move_value > alpha:
                    alpha = cur_move_value
                    best_move = successor_state

            # All moves considered result in us getting captures, 
            # pick a random move and continue to not waste time
            if best_move is None and successor_states is not None:
                best_move = successor_states[0]

            # set our move to be returned
            move = ("PLACE", best_move.move[0], best_move.move[1])
            self.last_search_time = time.process_time() - start_search_time

        # Explicitly type casting as suggested by Alexander Zable in Ed Thread #118
        if move[0] == "PLACE":
            move = (str(move[0]), int(move[1]), int(move[2]))

        # update time elapsed
        self.time_elapsed += time.process_time() - action_start_time

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
        turn_start_time = time.process_time()
        # store player colour using our own defined constants
        if player == "red":
            player_colour = tenuOS.util.constants.RED
        elif player == "blue":
            player_colour = tenuOS.util.constants.BLUE

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
            self.board_state[q][r] = tenuOS.util.constants.EMPTY
            # store the coords where the reflected tile will be placed
            self.steal_coords = (r, q)

        # calculate state after move applied
        next_state = SuccessorState(self.board_state, player_colour, self.board_size, self.num_tiles, (r, q))
        # extract state and number of tiles
        self.board_state = next_state.state
        self.num_tiles = next_state.num_tiles
        # store current tile difference for use with forward pruning of branches where we blunder captures
        self.tile_difference_threshold = self.tile_difference(self.board_state)
        # set max branching factor based on how filled the board is
        self.max_branching_factor = self.TILE_COUNT - self.num_tiles

        # allow for higher depth during endgames
        if self.max_branching_factor <= tenuOS.util.constants.HIGH_DEPTH_THRESHOLD:
            self.depth_limit = tenuOS.util.constants.HIGH_DEPTH
        elif self.max_branching_factor <= tenuOS.util.constants.MID_DEPTH_THRESHOLD:
            self.depth_limit = tenuOS.util.constants.MID_DEPTH
        else:
            self.depth_limit = tenuOS.util.constants.LOW_DEPTH

        # update knowledge of last played move for us or opponent
        if player_colour == self.player_colour:
            self.my_last_move = action
        else:
            self.opponent_last_move = action

        # update counter of moves played
        self.current_turn += 1
        self.time_elapsed += time.process_time() - turn_start_time

    def cutoff_test(self, successor_state, depth):
        """
        Test a hypothetical state at a given depth and determine if either
        the terminal state has been reached or the depth limit has been reached.
        If the terminal state has been reached, return either inf for us winning
        or -inf for the opponent winning. If the depth limit reached, return
        evaluation of the state.
        """

        # only check if at least board_size tiles exist and 2 * board_size - 1
        # moves have been played, as these are the minimum values for both
        # in a terminal state
        if successor_state.num_tiles >= self.board_size and self.current_turn >= 2 * self.board_size - 1:
            # calculate if state is terminal (someone won)
            terminal = self.terminal_state_check(successor_state)
            if terminal:
                # they win for even depth, we win for odd depth
                return -inf if depth % 2 == 0 else inf

        # cutoff depth reached, return evaluation
        if depth == self.depth_limit:
            return self.eval_func(successor_state.state)

        return None

    def terminal_state_check(self, successor_state):
        """
        Checks if the most recently played move won the game for the player
        who played the move.
        """

        # set the 2 edges to the colour of player who most recently played a move
        start_edge = (BoardEdge.BLUE_START if successor_state.player_colour ==
                                              tenuOS.util.constants.BLUE else BoardEdge.RED_START)
        goal_edge = (BoardEdge.BLUE_END if successor_state.player_colour ==
                                           tenuOS.util.constants.BLUE else BoardEdge.RED_END)
    
        start_path = search_path(successor_state.state, successor_state.player_colour, successor_state.board_size, successor_state.move, start_edge, Mode.WIN_TEST)
        end_path = search_path(successor_state.state, successor_state.player_colour, successor_state.board_size, successor_state.move, goal_edge, Mode.WIN_TEST)  

        # if player can reach both edges only traversing on their own colour tiles, they have won
        if start_path is not None and end_path is not None:
            return True

        return False

    def eval_func(self, state):
        """
        Evaluates a state, with a higher number meaning more payoff for our player.
        """

        # calculate all features
        win_dist_diff = self.win_distance_difference(state)
        tile_difference = self.tile_difference(state)
        two_bridge_count_diff = self.two_bridge_count_diff(state)

        # sum features according to weights
        evaluation = 0.7 * win_dist_diff + 0.7 * tile_difference - 0.4 * two_bridge_count_diff

        return evaluation

    def win_distance_difference(self, state):
        """
        Takes a state as input and calculates the 'win distance' for each colour,
        returns the win distance of the opponent colour minus the win distance of
        the self colour. Win distance is defined as the minimum number
        of empty tiles a colour needs to occupy to create a continuous path
        between both its edges, i.e. to win.
        """

        win_dists = {}
        edge_tile_freqs = {}

        # starting edges meet at (0, 0)
        # ending edges meet at (board_size - 1, board_size - 1)
        START_IND, END_IND = 0, self.board_size - 1

        # iterate over both colours, to calculate win distance for both
        for colour in (tenuOS.util.constants.RED, tenuOS.util.constants.BLUE):

            # using lists as wrappers to allow r and q to be modified by
            # assignment from dictionaries with r and q as values
            r, q = [0], [0]

            # dict to support iterating over r or q axes depending on colour
            axes = {tenuOS.util.constants.RED: r, tenuOS.util.constants.BLUE: q}

            # iterate over both the starting and ending edge for current colour
            # retrieving the 0th index to 'unwrap' and modify the index stored
            # in r or q
            for axes[colour][0] in (START_IND, END_IND):
                # store a dict of tile colour frequencies for both edges
                edge_tile_freqs[axes[colour][0]] = {tenuOS.util.constants.RED: 0, tenuOS.util.constants.BLUE: 0, tenuOS.util.constants.EMPTY: 0}
                # iterate over all tiles on the edge and store frequencies
                # of each colour
                for axes[(colour + 1) % 2][0] in range(self.board_size):
                    edge_tile_freqs[axes[colour][0]][state[r[0]][q[0]]] += 1

            # set the starting edge to the edge with the fewest opposite coloured tiles to minimise dijkstra calls
            if edge_tile_freqs[START_IND][(colour + 1) % 2] < edge_tile_freqs[END_IND][(colour + 1) % 2]:
                start_edge = BoardEdge.BLUE_START if colour == tenuOS.util.constants.BLUE else BoardEdge.RED_START
                goal_edge = BoardEdge.BLUE_END if colour == tenuOS.util.constants.BLUE else BoardEdge.RED_END
            else:
                start_edge = BoardEdge.BLUE_END if colour == tenuOS.util.constants.BLUE else BoardEdge.RED_END
                goal_edge = BoardEdge.BLUE_START if colour == tenuOS.util.constants.BLUE else BoardEdge.RED_START

            # if starting at start edge have init start node (0, 0)
            # if starting at end edge have init start node (0, board_size - 1)
            axes[colour][0] = START_IND if start_edge == (BoardEdge.BLUE_START if colour == tenuOS.util.constants.BLUE else BoardEdge.RED_START) else END_IND
            temp_path_cost = None
            # set win distance to an upperbound, if no path is found for one colour
            # that implies the other colour has won, will have already been captured
            # by terminal state checking
            win_dists[colour] = self.board_size * 2
            # iterate over starting edge
            for axes[(colour + 1) % 2][0] in range(self.board_size):
                # skip any tiles of opposite colour
                if state[r[0]][q[0]] == (colour + 1) % 2:
                    continue
                # call dijkstra and find the min path cost from that tile to the goal edge
                temp_path_cost = search_path(state, colour, self.board_size, (r[0], q[0]), goal_edge, Mode.WIN_DIST)
                # if no path is found do not update win_distance
                # if a path is found and is smaller than the current min win distance
                # update win distance to new minimum
                if temp_path_cost and temp_path_cost < win_dists[colour]:
                    win_dists[colour] = temp_path_cost

        # return win distance difference value such that higher is better for our colour
        win_dist_diff = win_dists[(colour + 1) % 2] - win_dists[colour]
        return win_dist_diff if self.player_colour == colour else -win_dist_diff

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
        Goes through the state and determines how many two-bridges the player and the opponent has and computes
        the difference between them. Used as an evaluation function feature. Two-bridges are vulnerable to captures
        and should ideally be avoided.
        """

        player_two_bridge_count = 0
        opponent_two_bridge_count = 0

        for r in range(self.board_size):
            for q in range(self.board_size):
                if state[r][q] != tenuOS.util.constants.EMPTY:
                    cur_cell_occupied_colour = state[r][q]
                    if cur_cell_occupied_colour == self.player_colour:
                        player_two_bridge_count += two_bridge_check("count", state, r, q,
                                                                    self.board_size, cur_cell_occupied_colour)
                    elif cur_cell_occupied_colour == (self.player_colour + 1) % 2:
                        opponent_two_bridge_count += two_bridge_check("count", state, r, q,
                                                                    self.board_size, cur_cell_occupied_colour)

        return player_two_bridge_count - opponent_two_bridge_count

    # -----------------------------------------------------------------------------------------------------------#
    # Pseudocode from lectures but with "game" variable omitted (though probably included through board_size and
    # player_colour)
    # -----------------------------------------------------------------------------------------------------------#
    def max_value(self, input_state, board_size, alpha, beta, depth, player_colour):
        """
        Max value function is called on states resulting from a move of the opponent's colour,
        calls min value function on states resulting from candidate moves by our player.
        """

        # if depth limit or terminal state reached, return evaluation
        # inf for if we win, -inf if we lose
        eval = self.cutoff_test(input_state, depth)
        if eval is not None:
            return eval

        successor_states = self.get_successor_states(input_state.state, board_size, input_state.num_tiles, player_colour, input_state.move, input_state.prev_move)

        for successor_state in successor_states:
            curr_succ_state_min_val = self.min_value(successor_state, board_size, alpha, beta, depth + 1,
                                                    (player_colour + 1) % 2)

            alpha = max(alpha, curr_succ_state_min_val)
            if alpha >= beta:
                return beta

        return alpha

    def min_value(self, input_state, board_size, alpha, beta, depth, player_colour):
        """
        Min value function is called on states resulting from a move of our own colour,
        calls max value function on states resulting from candidate moves by the opponent.
        """

        # if depth limit or terminal state reached, return evaluation
        # inf for if we win, -inf if we lose
        eval = self.cutoff_test(input_state, depth)
        if eval is not None:
            return eval

        # # Do forward pruning at depth 1 only if the depth_limit is 2
        # if depth == 1 and self.depth_limit == 2:
        #     if input_state.capture_prevention_check():
        #         return -(float('inf'))

        successor_states = self.get_successor_states(input_state.state, board_size, input_state.num_tiles, player_colour, input_state.move, input_state.prev_move)

        for successor_state in successor_states:
            curr_succ_state_max_val = self.max_value(successor_state, board_size, alpha, beta, depth + 1,
                                                    (player_colour + 1) % 2)

            beta = min(beta, curr_succ_state_max_val)
            if beta <= alpha:
                return alpha

        return beta

    def get_successor_states(self, state, board_size, num_tiles, player_colour, last_move, prior_move):
        """
        Takes a state as input, the player colour of whose turn it is in the given state,
        and returns a list of possible 'successor states' resulting from different moves
        that could be played.

        Generates successor states in rings about the tiles placed
        in the most recent two moves to optimise the order in which alpha-beta considers
        moves at depth 0, as the best move is likely to be near these tiles.
        """

        # list of successor states, dict to store which states have already been added
        successor_states = []
        added = {}

        # initialise dictionary to false for all tiles
        for r in range(board_size):
            for q in range(board_size):
                added[(r, q)] = False

        # dictionary constant used to iterate in spiral/circles about a central tile
        NEXT = {tenuOS.util.constants._30_DEG:  (-1, 1),
                tenuOS.util.constants._90_DEG:  (-1, 0),
                tenuOS.util.constants._150_DEG: (0 , -1),
                tenuOS.util.constants._210_DEG: (1 , -1),
                tenuOS.util.constants._270_DEG: (1 , 0),
                tenuOS.util.constants._330_DEG: (0 , 1)}
        # minimum num layers to explore about a tile
        MIN_LAYERS = 2
        # minimum states to return, else full board scan
        MIN_SUCCESSOR_STATES = 12

        # repeat state generation about most recent move for both players
        for move in (last_move, prior_move):

            # arbitrary start at 30 degrees north
            angle = tenuOS.util.constants._30_DEG
            # this equates to +1 in the r axis
            r, q = move[0] + 1, move[1]

            # search all tiles LAYERS about tile placed by move
            for layer in range(tenuOS.util.constants.MAX_LAYERS):
                # if after MIN_LAYERS layers, at least MIN_SUCCESSOR_STATES successor states
                # have been found, break
                if layer >= MIN_LAYERS and len(successor_states) >= MIN_SUCCESSOR_STATES:
                    break
                # repeat for each side of the layer (6 times as hexagons)
                for i in range(len(NEXT)):
                    # the number of cells on each edge == the layer
                    for j in range(layer):
                        # if tile is valid (within the board dimensions), not yet had its successor state added,
                        # and is currently empty, generate successor state and add to list
                        if is_valid_cell(r, q, board_size) and (not added[(r, q)]) and state[r][q] == tenuOS.util.constants.EMPTY:
                            # update added dict
                            added[(r, q)] = True
                            successor_states.append(SuccessorState(state, player_colour, board_size, num_tiles, (r, q), last_move))
                        # adjust r and q according the current angle
                        r += NEXT[angle][0]
                        q += NEXT[angle][1]
                    # move to next edge
                    angle = (angle + 1) % len(NEXT)

                layer += 1

            # if after MAX_LAYERS layers have been checked, less than MIN_SUCCESSOR_STATES
            # successor states have been found, do a full board scan and add states until
            # all states added or MIN_SUCCESSOR_STATES have been found total
            if len(successor_states) < MIN_SUCCESSOR_STATES:
                for r in range(board_size):
                    for q in range(board_size):
                        if len(successor_states) < MIN_SUCCESSOR_STATES and (not added[(r, q)]) and state[r][q] == tenuOS.util.constants.EMPTY:
                            successor_states.append(SuccessorState(state, player_colour, board_size, num_tiles, (r, q), last_move))

        return successor_states

class SuccessorState:

    def __init__(self, state, player_colour, board_size, num_tiles, move, prev_move=None):

        # generate copy of the state
        self.state = np.copy(state)
        # store the move being considered
        self.move = move
        # store the most recent move in the state parameter
        self.prev_move = prev_move
        # store the player colour who is playing the move being considered
        self.player_colour = player_colour
        # passing board size for use in methods
        self.board_size = board_size
        # storing number of tiles used, to be extracted for optimization
        self.num_tiles = num_tiles

        # apply the considered move to state instance variable
        self.apply_move()

    def apply_move(self):
        """
        Applies the move being considered to the state to determine the outcome state.
        """
        # Change the cell to player's colour
        self.state[self.move[0]][self.move[1]] = self.player_colour

        r = self.move[0]
        q = self.move[1]

        cells_to_remove = []

        # Check for possible captures and tag the cells to be removed from captures
        # ----------------------------------2-Bridge Capture Cases-------------------------------------------
        two_bridge_check("capture", self.state, r, q, self.board_size, self.player_colour, cells_to_remove)

        # -------------------------------------All Cells Adjacent Cases--------------------------------------------
        # Where the cells involved in a capture are all adjacent to this current move.
        all_cells_adjacent_check("capture", self.state, r, q, self.board_size, self.player_colour, cells_to_remove)

        # updating num_tiles
        self.num_tiles += 1 - len(cells_to_remove)

        for cell in cells_to_remove:
            self.state[cell[0]][cell[1]] = tenuOS.util.constants.EMPTY

    def capture_prevention_check(self):
        r = self.move[0]
        q = self.move[1]

        if (two_bridge_check("prevention", self.state, r, q, self.board_size, self.player_colour) or
            all_cells_adjacent_check("prevention", self.state, r, q, self.board_size, self.player_colour)):
            return True
        else:
            return False


"""
mode: A parameters which functions like a flag. Either "capture", "count" or "prevention" should be passed.
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
    if mode == "count":
        tile_status = tenuOS.util.constants.EMPTY
    else:
        tile_status = (player_colour + 1) % 2

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
                if mode == "prevention":
                    if state[r + 1][q - 1] == tile_status or state[r + 1][q] == tile_status:
                        return True
                else:
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
                if mode == "prevention":
                    if state[r + 1][q - 1] == tile_status or state[r][q - 1] == tile_status:
                        return True
                else:
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
                if mode == "prevention":
                    if state[r][q + 1] == tile_status or state[r + 1][q] == tile_status:
                        return True
                else:
                    if state[r][q + 1] == tile_status and state[r + 1][q] == tile_status:
                        if mode == "capture":
                            cells_to_remove.append((r, q + 1))
                            cells_to_remove.append((r + 1, q))
                        elif mode == "count":
                            two_bridge_count += 1

    # Only check the bottom cases when checking it's not counting for 2-bridges
    if mode != "count":
        # If there is an occupied cell of the same colour distance 2 away to the bottom-left of this current move.
        if is_valid_cell(r - 1, q - 1, board_size):
            if state[r - 1][q - 1] == player_colour:
                # Then, if there's also occupied cells belonging to the opponent distance 1 to the left and bottom left
                # of this move, this move is a capture.
                if is_valid_cell(r, q - 1, board_size) and is_valid_cell(r - 1, q, board_size):
                    if mode == "prevention":
                        if state[r][q - 1] == tile_status or state[r - 1][q] == tile_status:
                            return True
                    else:
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
                    if mode == "prevention":
                        if state[r - 1][q] == tile_status or state[r - 1][q + 1] == tile_status:
                            return True
                    else:
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
                    if mode == "prevention":
                        if state[r - 1][q + 1] == tile_status or state[r][q + 1] == tile_status:
                            return True
                    else:
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

"""
mode: A parameters which functions like a flag. Either "capture" or "prevention" should be passed.
state: A parameter which is the board state to evaluate. Either a 2D numpy array (what we use) or a 2D Python list.
r: r-coordinate of the move to be evaluated in a capture or two-bridge count.
q: q-coordinate of the move to be evaluated in a capture or two-bridge count.
board_size: The size of the board the Cachex game is being played on.
player_colour: The player_colour of the move to check.
cells_to_remove: Optional parameter; used with "capture". A list of the moves to be removed that is passed in by the
                 calling function and returned to them.
"""
def all_cells_adjacent_check(mode, state, r, q, board_size, player_colour, cells_to_remove = None):
    opponent_colour = (player_colour + 1) %2

    # If there is an occupied cell of the same colour distance 1 to the left of this current move
    if is_valid_cell(r, q - 1, board_size):
        if state[r][q - 1] == player_colour:
            # Then, if there's also occupied cells belonging to the opponent distance 1 to the top left and
            # bottom left of this move, this move is a capture.
            if is_valid_cell(r + 1, q - 1, board_size) and is_valid_cell(r - 1, q, board_size):
                if mode == "prevention":
                    if state[r + 1][q - 1] == opponent_colour or state[r - 1][q] == opponent_colour:
                        return True
                else:
                    if state[r + 1][q - 1] == opponent_colour and state[r - 1][q] == opponent_colour:
                        cells_to_remove.append((r + 1, q - 1))
                        cells_to_remove.append((r - 1, q))

    # If there is an occupied cell of the same colour distance 1 to the right of this current move
    if is_valid_cell(r, q + 1, board_size):
        if state[r][q + 1] == player_colour:
            # Then, if there's also occupied cells belonging to the opponent distance 1 to the top right and
            # bottom right of this move, this move is a capture.
            if is_valid_cell(r + 1, q, board_size) and is_valid_cell(r - 1, q + 1, board_size):
                if mode == "prevention":
                    if state[r + 1][q] == opponent_colour or state[r - 1][q + 1] == opponent_colour:
                        return True
                else:
                    if state[r + 1][q] == opponent_colour and state[r - 1][q + 1] == opponent_colour:
                        cells_to_remove.append((r + 1, q))
                        cells_to_remove.append((r - 1, q + 1))

    # If there is an occupied cell of the same colour distance 1 to the top left of this current move
    if is_valid_cell(r + 1, q - 1, board_size):
        if state[r + 1][q - 1] == player_colour:
            # Then, if there's also occupied cells belonging to the opponent distance 1 to the left and top right
            # of this move, this move is a capture.
            if is_valid_cell(r, q - 1, board_size) and is_valid_cell(r + 1, q, board_size):
                if mode == "prevention":
                    if state[r][q - 1] == opponent_colour or state[r + 1][q] == opponent_colour:
                        return True
                else:
                    if state[r][q - 1] == opponent_colour and state[r + 1][q] == opponent_colour:
                        cells_to_remove.append((r, q - 1))
                        cells_to_remove.append((r + 1, q))

    # If there is an occupied cell of the same colour distance 1 to the bottom right of this current move
    if is_valid_cell(r - 1, q + 1, board_size):
        if state[r - 1][q + 1] == player_colour:
            # Then, if there's also occupied cells belonging to the opponent distance 1 to the bottom left and right
            # of this move, this move is a capture.
            if is_valid_cell(r - 1, q, board_size) and is_valid_cell(r, q + 1, board_size):
                if mode == "prevention":
                    if state[r - 1][q] == opponent_colour or state[r][q + 1] == opponent_colour:
                        return True
                else:
                    if state[r - 1][q] == opponent_colour and state[r][q + 1] == opponent_colour:
                        cells_to_remove.append((r - 1, q))
                        cells_to_remove.append((r, q + 1))

    # If there is an occupied cell of the same colour distance 1 to the bottom left of this current move
    if is_valid_cell(r - 1, q, board_size):
        if state[r - 1][q] == player_colour:
            # Then, if there's also occupied cells belonging to the opponent distance 1 to the left and bottom right
            # of this move, this move is a capture.
            if is_valid_cell(r, q - 1, board_size) and is_valid_cell(r - 1, q + 1, board_size):
                if mode == "prevention":
                    if state[r][q - 1] == opponent_colour or state[r - 1][q + 1] == opponent_colour:
                        return True
                else:
                    if state[r][q - 1] == opponent_colour and state[r - 1][q + 1] == opponent_colour:
                        cells_to_remove.append((r, q - 1))
                        cells_to_remove.append((r - 1, q + 1))

    # If there is an occupied cell of the same colour distance 1 to the top right of this current move
    if is_valid_cell(r + 1, q, board_size):
        if state[r + 1][q] == player_colour:
            # Then, if there's also occupied cells belonging to the opponent distance 1 to the top left and right
            # of this move, this move is a capture.
            if is_valid_cell(r + 1, q - 1, board_size) and is_valid_cell(r, q + 1, board_size):
                if mode == "prevention":
                    if state[r + 1][q - 1] == opponent_colour or state[r][q + 1] == opponent_colour:
                        return True
                else:
                    if state[r + 1][q - 1] == opponent_colour and state[r][q + 1] == opponent_colour:
                        cells_to_remove.append((r + 1, q - 1))
                        cells_to_remove.append((r, q + 1))