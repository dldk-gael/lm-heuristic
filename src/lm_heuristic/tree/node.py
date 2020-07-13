"""
Define an abstract class from which all tree must inherate in order to be able to perform
a tree search on int
"""

from abc import ABC, abstractmethod
from typing import *
import random


class Node(ABC):
    """
    The node abstract object specifies the methods that needs to be implemented in order to 
    used a custom node object with the different tree search srategies defined in the tree_search modules. 
    """
    def __init__(self):
        ...

    @abstractmethod
    def is_terminal(self) -> bool:
        ...

    @abstractmethod
    def children(self) -> List["Node"]:
        ...

    def random_children(self) -> "Node":
        assert not self.is_terminal(), "Try to access children of a terminal node :%s" % str(self)
        return random.choice(self.children())

    def random_walk(self, debug=False) -> "Node":
        node = self
        while node is not None and not node.is_terminal():
            if debug:
                print(node)
            node = node.random_children()
        if debug:
            print(node)
        return node

    @abstractmethod
    def __str__(self):
        ...

    @abstractmethod
    def __hash__(self):
        ...

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __repr__(self):
        return str(self)