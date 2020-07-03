from abc import ABC, abstractmethod
from typing import *


class Node(ABC):
    """
    Abstract class to implement a node class that can be used in tree-based algorithm
    """

    def __init__(self):
        ...

    @abstractmethod
    def is_terminal(self) -> bool:
        """
        Return true if the current node is a leaf
        """
        ...

    @abstractmethod
    def children(self) -> List["Node"]:
        """
        Return the list of all children nodes from current node
        """
        ...

    @abstractmethod
    def random_children(self) -> "Node":
        """
        return a random children
        """
        ...

    def random_walk(self) -> "Node":
        """
        Perform a random walk from current node to a terminal node
        and return a leaf.
        Can return None if no terminal node can be access from current node
        (for example when using feature grammar)
        """
        node = self
        while node is not None and not node.is_terminal():
            node = node.random_children()
        return node

    @abstractmethod
    def __str__(self):
        """
        All node must be represented as a string
        """
        ...

    @abstractmethod
    def __hash__(self):
        ...

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()
