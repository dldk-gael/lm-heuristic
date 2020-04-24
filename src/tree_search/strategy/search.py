from abc import ABC, abstractmethod
from tree_search.tree import Node


class TreeSearch(ABC):
    def __init__(self, root: Node, **kwargs):
        pass

    @abstractmethod
    def search(self) -> Node:
        pass

