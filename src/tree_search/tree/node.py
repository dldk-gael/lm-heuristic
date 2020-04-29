from abc import ABC, abstractmethod


class Node(ABC):
    """
    Abstract class to implement a node class that can be used in tree-based algorithm
    """
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def is_terminal(self):
        """
        Return true if the current node is a leaf
        """
        pass

    @abstractmethod
    def childrens(self):
        """
        Return the list of all children nodes from current node
        """
        pass

    @abstractmethod
    def random_children(self):
        """
        return a random children
        """
        pass

    def random_walk(self):
        node = self
        while not node.is_terminal():
            node = node.random_children()
        return node

    @abstractmethod
    def __str__(self):
        pass