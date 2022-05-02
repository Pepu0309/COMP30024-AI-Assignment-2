from cmath import inf
from util.enums import *
from dijkstra.pathfinding import search_path
from util.general import *
import copy

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
            self.player_colour = Tile.RED
        elif player == "blue":
            self.player_colour = Tile.BLUE

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

        #print("state at start of action()")
        #print_state(self.board_state)

        best_move = None
        for successor_state in self.get_successor_states(self.board_state.copy(), self.board_size, self.player_colour):
            # For each move their, evaluation function value should be the minimum value of its successor states
            # due to game theory (opponent plays the lowest value move). Hence, we call min_value for all
            # the potential moves available to us in this current turn.
            cur_move_value = self.min_value(successor_state.state, self.board_size, alpha, beta, 1, self.player_colour)

            if cur_move_value > alpha:
                alpha = cur_move_value
                best_move = successor_state

        #print(best_move)

        return ("PLACE", best_move.move_r, best_move.move_q)
    
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
        
        next_state = SuccessorState(self.board_state.copy(), r, q, ENUM_CONVERSIONS[player], self.board_size)
        self.board_state = next_state.state.copy()

        #print("state at end of turn()")
        #print_state(self.board_state)

        # Steal doesn't have r and q, need to store previous move or something, will deal with it in action.

    def get_successor_states(self, state, board_size, player_colour):
        # Create successor for the moves using the player colour
        successor_states = []
        
        #print("get_succ_state call")
        #print_state(state)
        for r in range(board_size):
            for q in range(board_size):
                if state[r][q] == Tile.EMPTY:
                    new_state = SuccessorState(copy.deepcopy(state), r, q, player_colour, board_size)
                    successor_states.append(new_state)
            
        #print_state(state)
        return successor_states

    def cutoff_test(self, state, depth):
        # cutoff_depth or terminal_state
        if depth == 3:
            return True

        return False

    def eval_func(self, state):

        #print("state at start of eval()")
        #print_state(state)
        
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
                    return BoardEdge.RED_END
                elif tile_dict_end[Tile.BLUE] - tile_dict_start[Tile.BLUE]:
                    return BoardEdge.RED_START
                else:
                    if tile_dict_start[Tile.RED] > tile_dict_end[Tile.RED]:
                        return BoardEdge.RED_START
                    else:
                        return BoardEdge.RED_END

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
                    return BoardEdge.BLUE_END
                elif tile_dict_end[Tile.RED] - tile_dict_start[Tile.RED]:
                    return BoardEdge.BLUE_START
                else:
                    if tile_dict_start[Tile.BLUE] > tile_dict_end[Tile.BLUE]:
                        return BoardEdge.BLUE_START
                    else:
                        return BoardEdge.BLUE_END

            # if starting at start edge have init start node (0, 0)
            # if starting at end edge have init start node (0, board_size - 1)
            starting_edge = starting_edge_blue(state)
            q = 0 if starting_edge == BoardEdge.BLUE_START else board_size - 1
            goal_edge = BoardEdge.BLUE_END if starting_edge == BoardEdge.BLUE_START else BoardEdge.BLUE_START
            min_win_dist_blue = board_size * 2
            for r in range(board_size):
                if state[r][q] == Tile.RED: pass
                temp_path_cost = search_path(state, Tile.BLUE, board_size, (r, q), goal_edge, Mode.WIN_DIST)
                if temp_path_cost and temp_path_cost < min_win_dist_blue:
                    min_win_dist_blue = temp_path_cost
            
            # if starting at start edge have init start node (0, 0)
            # if starting at end edge have init start node (0, board_size - 1)
            starting_edge = starting_edge_red(state)
            r = 0 if starting_edge == BoardEdge.RED_START else board_size - 1
            goal_edge = BoardEdge.RED_END if starting_edge == BoardEdge.RED_START else BoardEdge.RED_START
            min_win_dist_red = board_size * 2
            for q in range(board_size):
                if state[r][q] == Tile.BLUE: pass
                temp_path_cost = search_path(state, Tile.RED, board_size, (r, q), goal_edge, Mode.WIN_DIST)
                if temp_path_cost and temp_path_cost < min_win_dist_red:
                    min_win_dist_red = temp_path_cost

            #print(min_win_dist_blue)
            #print(min_win_dist_red)
            # return win distance difference value such that higher is better for our colour
            win_dist_diff = min_win_dist_red - min_win_dist_blue
            return win_dist_diff if player_colour == Tile.BLUE else -win_dist_diff

        win_dist_diff = win_distance_difference(state, self.board_size, self.player_colour)
        evaluation = 1 * win_dist_diff
        return evaluation


    # Pseudocode from lectures but with "game" variable omitted (though probably included through board_size and
    # player_colour
    def max_value(self, state, board_size, alpha, beta, depth, player_colour):
        if self.cutoff_test(state, depth):
            return self.eval_func(state)

        #print("state at start of max_value()")
        #print_state(state)

        successor_states = self.get_successor_states(state.copy(), board_size, player_colour)
        for successor_state in successor_states:
            alpha = max(alpha, self.min_value(successor_state.state, board_size, alpha, beta, depth+1, Tile((player_colour.value + 1) % 2)))
            if alpha >= beta:
                return beta

        return alpha


    def min_value(self, state, board_size, alpha, beta, depth, player_colour):
        if self.cutoff_test(state, depth):
            return self.eval_func(state)

        #print("state at start of min_value()")
        #print_state(state)

        successor_states = self.get_successor_states(state.copy(), board_size, player_colour)
        for successor_state in successor_states:
            beta = min(beta, self.max_value(successor_state.state, board_size, alpha, beta, depth+1, Tile((player_colour.value + 1) % 2)))
            if beta <= alpha:
                return alpha

        return beta

class SuccessorState:

    def __init__(self, state, move_r, move_q, player_colour, board_size):
        self.state = state.copy()
        self.move_r = move_r
        self.move_q = move_q
        self.player_colour = player_colour
        self.board_size = board_size

        #print("state before apply move")
        #print_state(state)
        self.apply_move()
        #print("state after apply move")
        #print_state(state)

    def apply_move(self):
        # Change the cell to player's colour
        self.state[self.move_r][self.move_q] = self.player_colour

        r = self.move_r
        q = self.move_q
        board_size = self.board_size

        cells_to_remove = []
        opponent_colour = Tile((self.player_colour.value + 1) % 2)

        # Check for possible captures and tag the cells to be removed from captures
        # ----------------------------------Opposite Colour Adjacent Cases-------------------------------------------
        # If there is an occupied cell of the same colour distance 2 away above this current move
        if is_valid_cell(r+2, q-1, board_size):
            if self.state[r+2][q-1] == self.player_colour:
                # Then, if there's also occupied cells belonging to the opponent distance 1 to the top left and
                # top right of this move, this move is a capture.
                if is_valid_cell(r + 1, q - 1, board_size) and is_valid_cell(r + 1, q, board_size):
                    if self.state[r+1][q-1] == opponent_colour and self.state[r+1][q] == opponent_colour:
                        cells_to_remove.append((r + 1, q - 1))
                        cells_to_remove.append((r + 1, q))

        # If there is an occupied cell of the same colour distance 2 away to the top-left of this current move.
        if is_valid_cell(r+1, q-2, board_size):
            if self.state[r+1][q-2] == self.player_colour:
                # Then, if there's also occupied cells belonging to the opponent distance 1 to the top left and left
                # of this move, this move is a capture.
                if is_valid_cell(r+1, q-1, board_size) and is_valid_cell(r, q-1, board_size):
                    if self.state[r+1][q-1] == opponent_colour and self.state[r][q-1] == opponent_colour:
                        cells_to_remove.append((r + 1, q - 1))
                        cells_to_remove.append((r, q - 1))

        # If there is an occupied cell of the same colour distance 2 away to the bottom-left of this current move.
        if is_valid_cell(r-1, q-1, board_size):
            if self.state[r-1][q-1] == self.player_colour:
                # Then, if there's also occupied cells belonging to the opponent distance 1 to the left and bottom left
                # of this move, this move is a capture.
                if is_valid_cell(r, q-1, board_size) and is_valid_cell(r-1, q, board_size):
                    if self.state[r][q-1] == opponent_colour and self.state[r-1][q]:
                        cells_to_remove.append((r, q - 1))
                        cells_to_remove.append((r - 1, q))

        # If there is an occupied cell of the same colour distance 2 away to the bottom of this current move
        if is_valid_cell(r-2, q+1, board_size):
            if self.state[r-2][q+1] == self.player_colour:
                # Then, if there's also occupied cells belonging to the opponent distance 1 to the bottom left and
                # bottom right of this move, then this move is a capture.
                if is_valid_cell(r-1, q, board_size) and is_valid_cell(r-1, q+1, board_size):
                    if self.state[r-1][q] == opponent_colour and self.state[r-1][q+1] == opponent_colour:
                        cells_to_remove.append((r - 1, q))
                        cells_to_remove.append((r - 1, q + 1))

        # If there is an occupied cell of the same colour distance 2 away to the bottom right of this current move
        if is_valid_cell(r-1, q+2, board_size):
            if self.state[r-1][q+2] == self.player_colour:
                # Then, if there's also occupied cells belonging to the opponent distance 1 to the bottom right and
                # right of this move, then this move is a capture.
                if is_valid_cell(r-1, q+1, board_size) and is_valid_cell(r, q+1, board_size):
                    if self.state[r-1][q+1] == opponent_colour and self.state[r][q+1] == opponent_colour:
                        cells_to_remove.append((r - 1, q + 1))
                        cells_to_remove.append((r, q + 1))

        # If there is an occupied cell of the same colour distance 2 away to the top right of this current move
        if is_valid_cell(r+1, q+1, board_size):
            if self.state[r+1][q+1] == self.player_colour:
                # Then, if there's also occupied cells belonging to the opponent distance 1 to the right and top right
                # of this move, then this move is a capture.
                if is_valid_cell(r, q+1, board_size) and is_valid_cell(r+1, q, board_size):
                    if self.state[r][q+1] == opponent_colour and self.state[r+1][q] == opponent_colour:
                        cells_to_remove.append((r, q + 1))
                        cells_to_remove.append((r + 1, q))

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
                        cells_to_remove.append((r + 1, q))

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
            self.state[cell[0]][cell[1]] = Tile.EMPTY
