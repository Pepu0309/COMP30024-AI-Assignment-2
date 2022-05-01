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
        self.player_colour = player
        self.board_state = []
        self.board_size = n
        for r in range(n):
            board_row = []
            self.board_state.append(board_row)
            for q in range(n):
                board_row.append("unoccupied")

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


def eval_func(state):
    # evaluation function goes here
    return 0

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

    def apply_move(self):
        r = self.move_r
        q = self.move_q

        cell_to_remove = []
        # Temporary placeholder, implement logic to determine opponent colour later
        opponent_colour = "red"
        # ADD IS_VALID CHECK LATER

        # ----------------------------------Opposite Colour Adjacent Cases-------------------------------------------
        # If there is an occupied cell of the same colour distance 2 away above this current move
        if self.state[r+2][q-1] == self.player_colour:
            # Then, if there's also occupied cells belonging to the opponent distance 1 to the top left and
            # top right of this move, this move is a capture.
            if self.state[r+1][q-1] == opponent_colour and self.state[r+1][q] == opponent_colour:
                cell_to_remove.append((r+1, q-1))
                cell_to_remove.append((r+1, q))

        # If there is an occupied cell of the same colour distance 2 away to the top-left of this current move.
        if self.state[r+1][q-2] == self.player_colour:
            # Then, if there's also occupied cells belonging to the opponent distance 1 to the top left and left
            # of this move, this move is a capture.
            if self.state[r+1][q-1] == opponent_colour and self.state[r][q-1] == opponent_colour:
                cell_to_remove.append((r+1, q-1))
                cell_to_remove.append((r, q - 1))

        # If there is an occupied cell of the same colour distance 2 away to the bottom-left of this current move.
        if self.state[r-1][q-1] == self.player_colour:
            # Then, if there's also occupied cells belonging to the opponent distance 1 to the left and bottom left
            # of this move, this move is a capture.
            if self.state[r][q-1] == opponent_colour and self.state[r-1][q]:
                cell_to_remove.append((r, q-1))
                cell_to_remove.append((r-1, q))

        # If there is an occupied cell of the same colour distance 2 away to the bottom of this current move
        if self.state[r-2][q+1] == self.player_colour:
            # Then, if there's also occupied cells belonging to the opponent distance 1 to the bottom left and
            # bottom right of this move, then this move is a capture.
            if self.state[r-1][q] == opponent_colour and self.state[r-1][q+1] == opponent_colour:
                cell_to_remove.append((r-1, q))
                cell_to_remove.append((r-1, q+1))

        # If there is an occupied cell of the same colour distance 2 away to the bottom right of this current move
        if self.state[r-1][q+2] == self.player_colour:
            # Then, if there's also occupied cells belonging to the opponent distance 1 to the bottom right and right
            # of this move, then this move is a capture.
            if self.state[r-1][q+1] == opponent_colour and self.state[r][q+1] == opponent_colour:
                cell_to_remove.append((r-1, q+1))
                cell_to_remove.append((r, q+1))

        # If there is an occupied cell of the same colour distance 2 away to the top right of this current move
        if self.state[r+1][q+1] == self.player_colour:
            # Then, if there's also occupied cells belonging to the opponent distance 1 to the right and top right
            # of this move, then this move is a capture.
            if self.state[r][q+1] == opponent_colour and self.state[r+1][q] == opponent_colour:
                cell_to_remove.append((r, q+1))
                cell_to_remove.append((r+1, q))

        # -------------------------------------Same Colour Adjacent Cases--------------------------------------------
        # If there is an occupied cell of the same colour distance 1 to the left of this current move
        if self.state[r][q-1] == self.player_colour:
            # Then, if there's also occupied cells belonging to the opponent distance 1 to the top left and
            # bottom left of this move, this move is a capture.
            if self.state[r+1][q-1] == opponent_colour and self.state[r-1][q] == opponent_colour:
                cell_to_remove.append((r+1, q-1))
                cell_to_remove.append((r-1, q))

        # If there is an occupied cell of the same colour distance 1 to the right of this current move
        if self.state[r][q+1] == self.player_colour:
            # Then, if there's also occupied cells belonging to the opponent distance 1 to the top right and
            # bottom right of this move, this move is a capture.
            if self.state[r+1][q] == opponent_colour and self.state[r-1][q+1] == opponent_colour:
                cell_to_remove.append((r+1, q))
                cell_to_remove.append((r-1, q+1))

        # If there is an occupied cell of the same colour distance 1 to the top left of this current move
        if self.state[r+1][q-1] == self.player_colour:
            # Then, if there's also occupied cells belonging to the opponent distance 1 to the left and top right
            # of this move, this move is a capture.
            if self.state[r][q-1] == opponent_colour and self.state[r+1][q] == opponent_colour:
                cell_to_remove.append((r, q-1))
                cell_to_remove.append((r+1, q))

        # If there is an occupied cell of the same colour distance 1 to the bottom right of this current move
        if self.state[r-1][q+1] == self.player_colour:
            # Then, if there's also occupied cells belonging to the opponent distance 1 to the bottom left and right
            # of this move, this move is a capture.
            if self.state[r-1][q] == opponent_colour and self.state[r][q+1] == opponent_colour:
                cell_to_remove.append((r-1, q))
                cell_to_remove.append((r+1, q))

        # If there is an occupied cell of the same colour distance 1 to the bottom left of this current move
        if self.state[r-1][q] == self.player_colour:
            # Then, if there's also occupied cells belonging to the opponent distance 1 to the left and bottom right
            # of this move, this move is a capture.
            if self.state[r][q-1] == opponent_colour and self.state[r-1][q+1] == opponent_colour:
                cell_to_remove.append((r, q-1))
                cell_to_remove.append((r-1, q+1))

        # If there is an occupied cell of the same colour distance 1 to the top right of this current move
        if self.state[r+1][q] == self.player_colour:
            # Then, if there's also occupied cells belonging to the opponent distance 1 to the top left and right
            # of this move, this move is a capture.
            if self.state[r+1][q-1] == opponent_colour and self.state[r][q+1] == opponent_colour:
                cell_to_remove.append((r+1, q-1))
                cell_to_remove.append((r, q+1))
