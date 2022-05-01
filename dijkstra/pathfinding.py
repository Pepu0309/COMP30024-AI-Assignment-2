# standard imports
from cmath import inf
import heapq

from util import print_coordinate
from tenuOS.enums import *


class Node:
    """
    Class for a graph node, where the node represents a candidate tile for
    a path across cachex board.

    Duplicate Node objects may exist during dijkstra or BFS search and graph
    traversal utilises constrains of cachex board whereby each Node can have
    upto 6 adjacent nodes, with edge weight 0 or 1 depending on use case.
    """
    def __init__(self, coords, state):
        self.r = coords[0]
        self.q = coords[1]
        self.colour = state[self.r][self.q]

    def __eq__(self, other):
        return self.r == other.r and self.q == other.q

    def __hash__(self):
        return hash((self.r, self.q))

    def get_adjacent_nodes(self, state):
        """
        Returns a list of 6 node objects, each corresponding to the tiles on the
        cachex board adjacent to the tile associated with the node object.
        Does not exclude nodes whose coordinates are outside board boundaries,
        as that is handled later with the tile_unavailable method.
        """
        adjacent_nodes = []
        # Going diagonally towards bottom left
        adjacent_nodes.append(Node((self.r - 1, self.q), state))
        # Going diagonally towards top right
        adjacent_nodes.append(Node((self.r + 1, self.q), state))
        # Going horizontally towards left
        adjacent_nodes.append(Node((self.r, self.q - 1), state))
        # Going horizontally towards right
        adjacent_nodes.append(Node((self.r, self.q + 1), state))
        # Going diagonally towards top left
        adjacent_nodes.append(Node((self.r + 1, self.q - 1), state))
        # Going diagonally towards bot right
        adjacent_nodes.append(Node((self.r - 1, self.q + 1), state))
        return adjacent_nodes

    def tile_unavailable(self, colour, mode, n):
        """
        Returns true if, for the given use case as defined by mode, the
        current node can be traversed while graph is being searched.
        """
        if self.out_of_bounds(n): 
            return True
        if (mode == Mode.EVAL and self.colour != Tile.EMPTY and self.colour != colour):
            return True
        elif (mode == Mode.WIN_TEST and self.colour != colour):
            return True
        return False        

    def out_of_bounds(self, n):
        """
        Returns True if the node is outside cachex board boundaries.
        """
        return not ((0 <= self.r < n) and (0 <= self.q < n))

    def print_node_coords(self):
        """
        Prints the coordinate of the node.
        """
        print_coordinate(self.r, self.q)


class NodeCost:
    """
    Stores a Node and its cumulative path cost.
    """
    def __init__(self, node, cumul_path_cost):
        self.node = node
        self.cumul_path_cost = cumul_path_cost

    def __lt__(self, other):
        """
        Defined to allow NodeCost objects to be sorted within priorty queue.
        """
        return self.cumul_path_cost < other.cumul_path_cost

    def adjacent_cost(self, colour, mode):
        """
        Calculated the cumulative path cost of adjacent nodes for the given
        mode, assuming path goes via self's node.
        """
        if mode == Mode.EVAL:
            if self.node.colour == Tile.EMPTY:
                return self.cumul_path_cost + 1
            elif self.node.colour == colour:
                return self.cumul_path_cost
            else:
                # should never be reached as node should not be expanded due
                # to tile_unavailable check
                return inf
        elif mode == Mode.WIN_TEST:
            if self.node.colour == colour:
                return 0
            else:
                # should never be reached as node should not be expanded due
                # to tile_unavailable check
                return inf

class PriorityQueue:
    """
    Priority queue class which stores objects in a heap and calls __lt__
    method for comparison
    """
    def __init__(self, type):
        self.heap = []
        self.type = type

    def insert_obj(self, obj):
        """
        Method for inserting object into PQ, must be of certain type.
        """
        if isinstance(obj, self.type):
            heapq.heappush(self.heap, obj)
        else:
            #error handling tbd
            pass

    def pop_min(self):
        """
        Method for popping minimum object as defined by __lt__ operator.
        """
        return heapq.heappop(self.heap)

    def is_empty(self):
        """
        Returns True if PQ is empty.
        """
        return len(self.heap) == 0


def search_path(successor_state, board_size, start_coords, goal_edge, mode):
    """
    Dijkstra's algorithm implementation for finding the shortest path from some
    starting tile to either of the four board edges. Adapted from an A*
    implementation focused on reaching specific goal cells rather than an entire
    edge of the cachex board, originally adapted from
    https://www.redblobgames.com/pathfinding/a-star/introduction.html#astar
    Uses modifications such as using a heap to implement PriorityQueue as well
    as Node and NodeCost classes. There also has been modifications made using
    the rules from the Cachex specification.

    This function can be called with one of two modes as defined by an enum:

    Mode.WIN_TEST

    Checking if the most recent move won the game, by calling to both goal edges
    from the most recently placed tile, whereby only own colour tiles can be
    traversed with constant cost, i.e. the graph is unweighted and dijkstra's 
    will collapse to breadth first search.
    
    Mode.EVAL

    Calculating the minimum number of empty tiles a colour needs to fill in for
    a winning path, as a feature of eval() whereby empty tiles have path cost 1
    and own colour tiles have path cost 0.
    """
    start_node = Node(start_coords, successor_state.state)

    # defining nested function to check for terminal / goal state
    # probably a better way to do this
    """
    if goal == Goal.BLUE_START:
        def terminal_test(node):
            return node.q == 0
    elif goal == Goal.BLUE_END:
        def terminal_test(node):
            return node.q == board_size
    elif goal == Goal.RED_START:
        def terminal_test(node):
            return node.r == 0
    elif goal == Goal.RED_END:
        def terminal_test(node):
            return node.r == board_size
    """

    # temporary method of testing which doesnt hardcode one of the 4 tests
    def win_test(node):
        if goal_edge == GoalEdge.BLUE_START:
            return node.q == 0
        elif goal_edge == GoalEdge.BLUE_END:
            return node.q == board_size
        elif goal_edge == GoalEdge.RED_START:
            return node.r == 0
        elif goal_edge == GoalEdge.RED_END:
            return node.r == board_size

    # initialise starting NodeCost obj with path cost 0 and insert into new pq        
    start_node_cost = NodeCost(start_node, 0)
    pq = PriorityQueue(type(start_node_cost))
    pq.insert_obj(start_node_cost)

    # intialise path and path cost dicts
    came_from_dict = {start_node: None}
    cumulative_path_cost_dict = {start_node: 0}

    while not pq.is_empty():

        # pop node cost obj with smallest cumulative path cost
        curr_node_cost = pq.pop_min()

        # check if it is a goal node (a node on the goal edge), if so return
        # cost of the path
        if win_test(curr_node_cost.node):
            return curr_node_cost.cumul_path_cost

        # loop through all of the node / tile's adjacent nodes / tiles
        for adjacent_node in curr_node_cost.node.get_adjacent_nodes():

            # skip if a node cannot be traversed to, i.e. out of bounds, or
            # wrong colour depending on mode
            if curr_node_cost.node.tile_unavailable(successor_state.colour, mode, board_size):
                continue

            # calculate cost of traversing to adjacent node for given mode    
            new_cost = curr_node_cost.adjacent_cost(successor_state.colour, mode)

            # if this node isn't in cumulative_cost_dict, we have not visited it
            # yet. If the new_cost is less than the value in the
            # cumulative_cost_dict, that means we found a better route to this
            # node. In both cases, we choose to explore that node
            if adjacent_node not in cumulative_path_cost_dict or new_cost < cumulative_path_cost_dict[adjacent_node]:
                cumulative_path_cost_dict[adjacent_node] = new_cost

                # when a node is inserted into the priority queue, a NodeCost object corresponding
                # to it will be created and heap.heapify will use the __lt__ comparator method of
                # the object to do heapsort
                adj_node_cost = NodeCost(adjacent_node, new_cost)
                pq.insert_obj(adj_node_cost)
                came_from_dict[adjacent_node] = curr_node_cost.node

    # no path was found
    return None

    # dont think there is a usecase for the # of nodes in the path
    """
    path = []
    # If path was found from start node to goal edge, create a list of the nodes
    # traversed to reach the goal edge
    if path_found:
        # insert the the final node that belongs to the goal edge, after which
        # prepend the previous node in the path in a loop until starting node
        # is added, resulting in a list in order of traversal of the path
        curr_node = curr_node_cost.node
        while curr_node is not None:
            path.insert(0, curr_node)
            curr_node = came_from_dict[curr_node]

    # length of the path
    return len(path)
    """

