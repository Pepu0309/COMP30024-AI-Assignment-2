def is_valid_cell(r, q, board_size):
    return 0 <= r < board_size and 0 <= q < board_size

def print_state(state):
    """
    Prints the current state to console
    """
    print("\nstate as a grid: \n")
    for r in reversed(state):
        for q in r:
            print(q, end = ", ")
        print("")
    print("")