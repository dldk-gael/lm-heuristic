from tree_search.tree.node import Node


class Counter(Node):
    def __init__(self, identifiant, parent: Node = None, is_terminal=False):
        Node.__init__(self)
        self.identifiant = identifiant
        self.__childrens = []
        self.parent = parent
        self.is_terminal = is_terminal
        self.counter = 0
        self.mean_reward = 0

    def __hash__(self):
        return hash(self.identifiant)

    def childrens(self):
        return self.__childrens

    def is_terminal(self):
        return self.is_terminal

    def add_children(self, node):
        self.__childrens.append(node)

    def update(self, new_reward):
        self.counter += 1
        self.mean_reward = self.mean_reward * (self.counter - 1) / self.counter + new_reward / self.counter

    def most_visited_children(self):
        return max(self.childrens(), key=lambda counter_node: counter_node.counter)
