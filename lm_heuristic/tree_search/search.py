from abc import ABC, abstractmethod
from typing import *
from copy import copy
import pandas as pd
import seaborn as sns

from ..tree import Node
from ..heuristic import Heuristic
from ..utils.timer import Timer, timeit


class TreeSearch(ABC, Timer):
    """
    Abstract class that define a tree searcher
    The objective of a tree searcher is to find the leaf that maximise an evaluation function
    """

    def __init__(self, heuristic: Heuristic, buffer_size: int = 1):
        Timer.__init__(self)
        self._name = ""
        self.heuristic = copy(heuristic)
        self.buffer_size = buffer_size
        self.best_leaf: Node
        self.best_leaf_value: float

    def reset(self):
        """
        Reset the search memory, the heuristic memory and the timer 
        """
        self.best_leaf = None
        self.best_leaf_value = None
        self.heuristic.reset()
        self.reset_timer()

    @timeit
    def __call__(self, root: Node, nb_of_tree_walks: int) -> Tuple[Node, float]:
        """
        Launch the search + keep in memory informations about the search
        :param root: Node from which the search will start
        :param nb_of_tree_walks to perform in total
        :return: best leave that was found and its evaluation value
        """
        self.reset()
        self.best_leaf, self.best_leaf_value = self._search(root, nb_of_tree_walks)
        return self.best_leaf, self.best_leaf_value

    def print_search_info(self):
        """
        Print several informations about the last search performed
        """
        nb_leaves_evaluated = len(self.heuristic.history_of_terminal_nodes())
        nb_unique_leaves = len(set(self.heuristic.history_of_terminal_nodes()))
        total_time = self.time_spent()
        evaluation_time = self.heuristic.time_spent()

        print(
            "--- SEARCH RESULT ---\n"
            "LEAVE EVALUATION : \n"
            "%d leaves evaluation was performed\n"
            "%0.2f%% of leaves was evaluated multiples times\n"
            "\nTIMING : \n"
            "The search tooks %.2fs\n"
            "%.2f%%  of the time was spent on leave evaluation\n"
            "\nRESULTS : \n"
            "Best leaf that have been found: %s \n"
            "It has a score of %f"
            % (
                nb_leaves_evaluated,
                (nb_leaves_evaluated - nb_unique_leaves) / nb_leaves_evaluated * 100,
                total_time,
                evaluation_time / total_time * 100,
                str(self.best_leaf),
                self.best_leaf_value,
            )
        )

    def print_path(self):
        """
        Print the path from root to best leaf that was found
        """
        print("The following path was taken :")
        for i, node in enumerate(self.path()):
            print("%d: %s" % (i, str(node)))

    def plot_leaf_values_distribution(self):
        """
        Plot the leaf value distribution 
        """
        values = self.heuristic.history_of_values()
        assert values != [], "Try to plot leaf values distribution, but no search was performed yet"

        series_values = pd.Series(values, name="Leaf values")
        sns.set()
        sns.distplot(series_values, label=str(self))

    def set_name(self, name: str):
        self._name = name

    def __str__(self) -> str:
        return self._name

    @abstractmethod
    def _search(self, root: Node, nb_of_tree_walks: int) -> Tuple[Node, float]:
        """
        search and return the terminal node that maximise the evalution function and its value
        """
        ...

    @abstractmethod
    def path(self) -> List[Node]:
        """
        return path taken from root node to best terminal node that has been found
        """
        ...
