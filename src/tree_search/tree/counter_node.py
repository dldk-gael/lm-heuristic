from tree_search.tree.node import Node
import random


class Counter(Node):
    """

    """
    def __init__(self, reference_node, parent: Node = None):
        Node.__init__(self)
        self.reference_node = reference_node
        self.__childrens = None
        self.parent = parent
        self.is_terminal = True
        self.count = 0
        self.mean_reward = 0

    def __hash__(self):
        return hash(self.reference_node)

    def expand(self):
        self.__childrens = [Counter(children_node, parent=self) for children_node in self.reference_node.childrens()]
        self.is_terminal = False

    def childrens(self):
        return self.__childrens

    def is_terminal(self):
        return self.is_terminal

    def update(self, new_reward):
        self.count += 1
        self.mean_reward = self.mean_reward * (self.count - 1) / self.count + new_reward / self.count

    def most_visited_children(self):
        return max(self.childrens(), key=lambda counter_node: counter_node.counter)

    def random_children(self):
        return random.choice(self.childrens())
