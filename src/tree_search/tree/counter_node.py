from tree_search.tree.node import Node
import random
from typing import *


class CounterNode(Node):
    """
    Class to wrap a node object inside a counter object.

    In addition to usual node method, a counter will keep in memory the following information :
    - the nomber of times the node has been visited by a search strategy
    - the expected reward from this node
    - the top reward that has been obtained from this node
    - the square sum reward obtained so far from this node

    Moreover, in order to be able to back-propagate the information, the counter node keep in memory a reference
    to his parent node
    """

    def __init__(self, reference_node, parent=None):
        """
        Create a new counter node
        :param reference_node: reference toward the node object that is encapsulate
        :param parent: reference toward the parent counter node
        """
        Node.__init__(self)
        self.reference_node = reference_node
        self.__childrens = None
        self.parent = parent
        self.__is_terminal = True
        self.count = 0
        self.sum_rewards = 0
        self.sum_of_square_rewards = 0
        self.top_reward = 0
        self.top_leaf_node = (
            None  # Useful to analyse and debug MCTS, will be remove later
        )
        self.freeze = (
            False  # Useful to stop backprogation if choice has already been made
        )

        # Idea taken from 'Attacking SameGame using Monte-Carlo Tree Search', Klein
        # A node is completely solved either if its reference node is a terminal node
        # or if all its children are completely solved.
        self.solved = False

    def expand(self):
        """
        Expand a counter node :
        - generate and save in memory all childrens
        """
        assert (
            not self.reference_node.is_terminal()
        ), "Try to expand from a terminal node"
        self.__childrens = [
            CounterNode(children_node, parent=self)
            for children_node in self.reference_node.childrens()
        ]
        self.__is_terminal = False

    def childrens(self) -> List["CounterNode"]:
        return self.__childrens

    def is_terminal(self) -> bool:
        return self.__is_terminal

    def backpropagate(self, new_reward, leaf):
        """
        Given a new_reward update the average reward, sum of square rewards and top reward
        and backprogate the information to the parent node
        """
        if not self.freeze:
            self.sum_rewards += new_reward
            self.sum_of_square_rewards += new_reward ** 2

            if new_reward > self.top_reward:
                self.top_reward = new_reward
                self.top_leaf_node = leaf
            self.top_reward = max(self.top_reward, new_reward)

            if self.parent is not None:
                self.parent.backpropagate(new_reward, leaf)

    def top_children(self) -> "CounterNode":
        """
        Return the children that have the best top_reward value
        """
        assert not self.is_terminal(), "Try to access childrens of a terminal node"
        return max(self.__childrens, key=lambda child: child.top_reward)

    def random_children(self) -> "CounterNode":
        assert not self.is_terminal(), "Try to access childrens of a terminal node"
        return random.choice(self.childrens())

    def set_as_solved(self):
        """
        Set the counter node as solved
        If all his brothers are also solved, back-propagate this information to his parent
        """
        self.solved = True

        if (self.parent is not None) and (not self.parent.freeze):
            brothers = self.parent.childrens()
            for brother in brothers:
                if not brother.solved:
                    return
            self.parent.set_as_solved()

    def detailed_node_info(self):
        print(self)
        if not self.is_terminal():
            print("\t has %d children" % len(self.childrens()))
            for i, children in enumerate(self.childrens()):
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
