from sample_players import DataPlayer

import random

MAX_ITER = 150
EXPLORE_FACTOR = .05

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
                
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            self.queue.put(MonteCarloTreeSearch(state))
    


class MonteCarloTreeNode():
    def __init__(self, state, parent=None):
        
        self.state = state
        self.parent = parent

        self.total_reward = 0
        self.visits_count = 1
        
        self.children = []
        self.children_actions = []


    def add_child(self, new_state, action):
        child = MonteCarloTreeNode(new_state, self)
        self.children.append(child)
        self.children_actions.append(action)

    def update(self, reward):
        self.total_reward += reward
        self.visits_count += 1

    def fully_explored(self):
        return len(self.children_actions) == len(self.state.actions())


def MonteCarloTreeSearch(state):
    parent = MonteCarloTreeNode(state)
    if parent.state.terminal_test():
        return random.choice(state.actions())
    for n in range(MAX_ITER):
        child = MCTS_policy(parent)
        if child:
            reward = MCTS_reward(child.state)
            MCTS_backprop(child, reward)

    best_child_id = parent.children.index(MCTS_best_child(parent))
    return parent.children_actions[best_child_id]


def MCTS_policy(node):
    # if not fully explored then expand, take the best child otherwise 
    
    while not node.state.terminal_test():
        if not node.fully_explored():
            return MCTS_expand(node)

        node = MCTS_best_child(node)

    return node


def MCTS_expand(parent):
    for action in parent.state.actions():
        if action not in parent.children_actions:
            new_state = parent.state.result(action)
            parent.add_child(new_state, action)
            return parent.children[-1]


def MCTS_best_child(parent):
    #Find the child node with the best score.
    import math

    best_score = float("-inf")
    best_children = []
    for child in parent.children:
        exploit = child.total_reward / child.visits_count
        explore = (2. * math.log(parent.visits_count) / child.visits_count)**.5
        score = exploit + EXPLORE_FACTOR * explore
        if score == best_score:
            best_children.append(child)
        elif score > best_score:
            best_children = [child]
            best_score = score

    return random.choice(best_children)


def MCTS_reward(state):
    #Random search to retrieve the reward
   
    player = state.player()
    while not state.terminal_test():
        action = random.choice(state.actions())
        state = state.result(action)

    return -1 if state._has_liberties(player) else 1


def MCTS_backprop(node, reward):
    #Use the result to update information in the nodes on the path.

    while node:
        node.update(reward)
        node = node.parent
        reward *= -1
