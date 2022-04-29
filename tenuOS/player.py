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
        self.alpha = float('-inf')
        self.beta = float('inf')
        self.board = []
        for r in range(n):
            board_row = []
            self.board.append(board_row)
            for q in range(n):
                board_row.append("unoccupied")

    def action(self):
        """
        Called at the beginning of your turn. Based on the current state
        of the game, select an action to play.
        """
        # put your code here
    
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
            self.board[r][q] = player
        # Steal doesn't have r and q, need to store previous move or something, will deal with it in action.
