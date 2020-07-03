"""
Define an abstract class from which all tree must inherate in order to be able to perform
a tree search on int
"""

from abc import ABC, abstractmethod
from typing import *
import random


class Node(ABC):
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

    def random_children(self) -> "Node":
        assert not self.is_terminal(), "Try to access children of a terminal node :%s" % str(self)
        return random.choice(self.children())

    def random_walk(self) -> "Node":
        node = self
        while node is not None and not node.is_terminal():
            node = node.random_children()
        return node

    @abstractmethod
    def __str__(self):
        ...

    @abstractmethod
    def __hash__(self):
        ...

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()
