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
        self.is_terminal = True
        self.count = 0
        self.average_reward = 0
        self.sum_of_square_rewards = 0
        self.top_reward = 0

    def expand(self):
        """
        Expand a counter node :
        - generate and save in memory all childrens
        """
        self.__childrens = [Counter(children_node, parent=self) for children_node in self.reference_node.childrens()]
        self.is_terminal = False

    def childrens(self):
        return self.__childrens

    def is_terminal(self):
        return self.is_terminal

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
        return max(self.__childrens, key=lambda child: child.top_reward)

    def random_children(self):
        return random.choice(self.childrens())
