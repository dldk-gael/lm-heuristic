from tree_search.tree.node import Node
import random


class Counter(Node):
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
        self.average_reward = 0
        self.sum_of_square_rewards = 0
        self.top_reward = 0

    def expand(self):
        """
        Expand a counter node :
        - generate and save in memory all childrens
        """
        assert not self.reference_node.is_terminal(), 'Try to expand from a terminal node'
        self.__childrens = [Counter(children_node, parent=self) for children_node in self.reference_node.childrens()]
        self.__is_terminal = False

    def childrens(self):
        return self.__childrens

    def is_terminal(self):
        return self.__is_terminal

    def update_and_backpropagate(self, new_reward):
        """
        Given a new_reward update the average reward, sum of square rewards and top reward
        and backprogate the information to the parent node
        """
        self.count += 1
        self.average_reward = self.average_reward * (self.count - 1) / self.count + new_reward / self.count
        self.sum_of_square_rewards += new_reward ** 2
        self.top_reward = max(self.top_reward, new_reward)

        if self.parent is not None:
            self.parent.update_and_backpropagate(new_reward)

    def top_children(self):
        """
        Return the children that have the best top_reward value
        """
        assert not self.is_terminal(), 'Try to access childrens of a terminal node'
        return max(self.__childrens, key=lambda child: child.top_reward)

    def random_children(self):
        assert not self.is_terminal(), "Try to access childrens of a terminal node"
        return random.choice(self.childrens())

    def __str__(self):
        return ("Reference node : %s\n"
                "\t count: %d\n"
                "\t average_reward: %f\n"
                "\t top_reward: %f"
                ) % (str(self.reference_node), self.count, self.average_reward, self.top_reward)
