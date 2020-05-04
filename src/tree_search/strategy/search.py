from abc import ABC, abstractmethod
from typing import *
from time import time
import pandas as pd
import seaborn as sns
from tree_search.tree import Node


class TreeSearch(ABC):
    """
    Abstract class that define a tree searcher
    The objective of a tree searcher is to find the leaf that maximise an evaluation function

    This class will keep update different informations during the search :
    - time needed to perform the search
    - time spent to evaluate the leaves
    - values that has been computed
    """

    def __init__(self, evaluation_fn: Callable[[List[Node]], List[float]], **kwargs):
        """
        Initialize a tree searcher
        :param evaluation_fn:
        :param kwargs:
        """
        self._evaluation_fn = evaluation_fn
        self._total_time = 0
        self._evaluation_time = 0
        self._values = []
        self._keep_track = False

    def reset(self):
        self._total_time = 0
        self._evaluation_time = 0
        self._values = []

    def __call__(self, root: Node, nb_of_tree_walks: int) -> Node:
        """
        Launch the search + keep in memory informations about the search
        :param root: Node from which the search will start
        :param nb_of_tree_walks
        :return: best leave that was found
        """
        self.reset()
        self._keep_track = True
        begin_time = time()
        result = self._search(root, nb_of_tree_walks)
        self._total_time = time() - begin_time
        self._keep_track = False
        return result

    def _eval_node(self, nodes: List[Node]) -> List[float]:
        """
        Handle the call to the evaluation_fn + keep in memory informations about the call
        :param nodes: list of leaves
        :return: leaves' scores
        """
        begin_eval_time = time()
        results = self._evaluation_fn(nodes)
        if self._keep_track:
            self._evaluation_time += time() - begin_eval_time
            self._values += results
        return results

    def print_search_info(self):
        """
        Print several informations about the last search performed
        """
        path, best_leaf, best_value = self.search_info()
        print(
            "--- Search information ---\n"
            "%d tree walks was performed in %.1f s\n"
            "%.2f%%  of the time was spent on leave evaluation"
            % (
                len(self._values),
                self._total_time,
                (self._evaluation_time / self._total_time) * 100,
            )
        )

        print(
            "Best leaf that have been found: %s \n"
            "It has a score of %f" % (str(best_leaf), best_value)
        )

        print("The following path was taken :")
        for i, node in enumerate(path):
            print("%d: %s" % (i, str(node)))

    def plot_leaf_values_distribution(self):
        assert (
            self._values != []
        ), "Try to plot leaf values distribution, but no search was performed yet"

        values = pd.Series(self._values, name="Leaf values")
        sns.set()
        sns.distplot(values, label=str(self))

    @abstractmethod
    def _search(self, root: Node, nb_of_tree_walks: int) -> Node:
        """
        search and return the terminal node that maximise the evalution function
        """
        ...

    @abstractmethod
    def path(self) -> List[Node]:
        """
        return path taken from root node to best terminal node that has been found
        """
        ...

    @abstractmethod
    def __str__(self) -> str:
        ...

    @abstractmethod
    def search_info(self) -> Tuple[List[Node], Node, float]:
        """
        :return: tupple path taken, best leaf found, value of the best leaf
        """
        ...
