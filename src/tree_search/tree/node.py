from abc import ABC, abstractmethod
from typing import List


class Node(ABC):
    """
    Abstract class to implement a node class that can be used in tree-based algorithm
    """
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """
        Return true if the current node is a leaf
        """
        pass

    @abstractmethod
    def childrens(self) -> List['Node']:
        """
        Return the list of all children nodes from current node
        """
        pass

    @abstractmethod
    def __hash__(self):
        """
        All node ojbect must be hashable
        """
        pass

