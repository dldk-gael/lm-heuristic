from abc import ABC, abstractmethod
from tree_search.tree import Node
from typing import List


class TreeSearch(ABC):
    """
    Abstract class that define a tree search
    Given :
    - a starting node
    - an evaluation function : list of terminal_node -> list of scores

    A TreeSearch object will search for the terminal node that maximise this evalution function
    """
    def __init__(self, root: Node, evaluation_fn, **kwargs):
        self.root = root
        self.evaluation_fn = evaluation_fn

    @abstractmethod
    def search(self) -> Node:
        """
        search and return the terminal node that maximise the evalution function
        """
        pass

    @abstractmethod
    def path(self) -> List[Node]:
        """
        return path taken from root node to best terminal node that has been found
        """
        pass

    def __call__(self) -> Node:
        return self.search()

    def _eval_node(self, node: List[Node]) -> List[float]:
        return self.evaluation_fn(node)

    @abstractmethod
    def search_info(self):
        """
        :return: dict containing the following information
                    - time needed to perform the search
                    - path taken
                    - total number of tree walks
                    - best leaf found
                    - value of the best leaf
        """
        pass

    def print_search_info(self):
        info = self.search_info()
        print("--- Search information ---\n"
              "%d tree walks was performed in %.1f s" % (info['total_nb_of_walks'], info['time']))

        print("Best leaf that have been found: %s \n"
              "It has a score of %f" % (str(info['best_leaf']), info['best_leaf_value']))

        print("The following path was taken :")
        for i, node in enumerate(info['path']):
            print("%d: %s" % (i, str(node)))

