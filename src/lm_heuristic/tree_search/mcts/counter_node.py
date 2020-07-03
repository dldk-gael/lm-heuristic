"""
Define a node's wrapper that maintains several statistics needed during the MCTS search
"""

import random
from typing import *

from lm_heuristic.tree import Node


class CounterNode(Node):
    """
    Class to wrap a node (which is denote as the reference node) and maintain several statistics during the MCTS:
    - the nomber of times the node has been visited
    - the expected reward from this node
    - the top reward that has been obtained from this node
    - a reference to the leaf corresponding to this top reward (only used to make it more to debug)
    - the square sum reward obtained so far from this node

    Contrary to a vanilla node, the counter node keeps in memory a reference to his parent node in order
    to be able to backpropagate the information
    """

    def __init__(self, reference_node: Node, parent: "CounterNode" = None):
        Node.__init__(self)
        self.reference_node = reference_node
        self._children = None
        self.parent = parent
        self._is_terminal = True
        self.count = 0
        self.sum_rewards = 0
        self.sum_of_square_rewards = 0
        self.top_reward = 0

        # Useful to analyse and debug MCTS
        self.top_leaf_node = None

        # If true will prevent any backpropagation to the node's parent
        self.freeze = False

        # A node is completely solved either if its reference node is a terminal node
        # or if all its children are completely solved.
        self.solved = False

    def expand(self):
        assert not self.reference_node.is_terminal(), "Try to expand from a terminal node"
        self._children = [
            CounterNode(children_node, parent=self) for children_node in self.reference_node.children()
        ]
        self._is_terminal = False

    def children(self) -> List["CounterNode"]: #type:ignore
        assert self._children, "Try to access children "
        return self._children

    def is_terminal(self) -> bool:
        return self._is_terminal

    def backpropagate(self, new_reward, leaf):
        """
        Given a new_reward update the average reward, sum of square rewards and top reward
        and backprogate the information to the parent node
        """
        if self.freeze:
            return

        self.sum_rewards += new_reward
        self.sum_of_square_rewards += new_reward ** 2

        if new_reward > self.top_reward:
            self.top_reward = new_reward
            self.top_leaf_node = leaf

        if self.parent is not None:
            self.parent.backpropagate(new_reward, leaf)

    def top_child(self) -> "CounterNode":
        """
        Return the children that have the best top_reward value
        """
        assert not self.is_terminal(), "Try to access children of a terminal node :%s" % str(self.reference_node)
        return max(self._children, key=lambda child: child.top_reward)

    def most_visited_child(self) -> "CounterNode":
        return max(self._children, key=lambda child: child.count)

    def random_children(self) -> "CounterNode":
        assert not self.is_terminal(), "Try to access children of a terminal node :%s" % str(self.reference_node)
        return random.choice(self.children())

    def set_as_solved(self):
        """
        Set the counter node as solved
        If all his brothers are also solved, back-propagate the information to his parent
        """
        self.solved = True

        if (self.parent is not None) and (not self.parent.freeze):
            brothers = self.parent.children()
            for brother in brothers:
                if not brother.solved:
                    return
            self.parent.set_as_solved()

    def detailed_node_info(self):
        """
        Print the children of a given node 
        """
        print(self)
        if not self.is_terminal():
            print("\t has %d children" % len(self.children()))
            for i, children in enumerate(self.children()):
                print("\n-- children n°%d --" % i)
                print(str(children))

    def __str__(self):
        return (
            "Reference node : %s\n"
            "\t count: %d\n"
            "\t average_reward: %f\n"
            "\t top_reward: %f\n"
            "\t top_leaf_associated : %s\n"
            "\t is solved : %s"
        ) % (
            str(self.reference_node),
            self.count,
            self.sum_rewards / self.count,
            self.top_reward,
            str(self.top_leaf_node),
            "yes" if self.solved else "no",
        )

    def __hash__(self):
        raise Exception("Try to hash a counter node")
