from abc import ABC, abstractmethod
from tree_search.tree import Node
from typing import List


class TreeSearch(ABC):
    """
    Abstract class that define a tree search
    Given :
    - a starting node
    - an evaluation function : terminal_node -> score

    A TreeSearch object will search for the terminal node that maximise this evalution function
    """
    def __init__(self, root: Node, evaluation_fn, **kwargs):
        self.root = root
        self.evaluation_fn = evaluation_fn

    @abstractmethod
    def search(self) -> Node:
        """
        perform
        """
        pass

    @abstractmethod
    def path(self) -> List[Node]:
        pass

    def __call__(self) -> Node:
        return self.search()

    def _eval_node(self, node):
        return self.evaluation_fn(node)
