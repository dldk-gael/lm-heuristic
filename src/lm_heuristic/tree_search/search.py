from abc import ABC, abstractmethod
from typing import *
import pandas as pd
import seaborn as sns

from lm_heuristic.tree import Node
from lm_heuristic.heuristic import Heuristic
from lm_heuristic.utils.timer import Timer, timeit


class TreeSearch(ABC, Timer):
    """
    Abstract class that define a tree searcher
    The objective of a tree searcher is to find the leaf that maximise an evaluation function
    """

    def __init__(self, heuristic: Heuristic, name: str = ""):
        Timer.__init__(self)
        self._name = name
        self._heuristic = heuristic

    @timeit
    def __call__(self, root: Node, nb_of_tree_walks: int) -> Tuple[Node, float]:
        """
        Launch the search + keep in memory informations about the search
        :param root: Node from which the search will start
        :param nb_of_tree_walks to perform in total
        :return: best leave that was found and its evaluation value
        """
        self._heuristic.reset()
        self.reset_timer()
        self._search(root, nb_of_tree_walks)
        return self._heuristic.best_node_evaluated()

    def print_search_info(self):
        """
        Print several informations about the last search performed
        """
        total_time = self.time_spent()
        evaluation_time = self._heuristic.time_spent()
        best_node, best_value = self._heuristic.best_node_evaluated()

        print(
            "--- SEARCH RESULT ---\n"
            "LEAVE EVALUATION : \n"
            "%d tree walks were performed\n"
            "The evaluation function was called on %d leaves \n"
            "\nTIMING : \n"
            "The search tooks %.2fs\n"
            "%.2f%%  of the time was spent on leave evaluation\n"
            "\nRESULTS : \n"
            "Best leaf that have been found: %s \n"
            "It has a score of %.5f"
            % (
                len(self._heuristic._history),
                self._heuristic._eval_counter,
                total_time,
                evaluation_time / total_time * 100,
                str(best_node),
                best_value
            )
        )

    def plot_leaf_values_distribution(self):
        """
        Plot the leaf value distribution
        """
        values = self._heuristic.history_of_values()
        assert values != [], "Try to plot leaf values distribution, but no search was performed yet"

        series_values = pd.Series(values, name="Leaf values")
        sns.set()
        sns.distplot(series_values, label=str(self))

    def set_name(self, name: str):
        self._name = name

    def __str__(self) -> str:
        return self._name

    def top_n_leaves(self, top_n: int = 1) -> List[Tuple[Node, float]]:
        return self._heuristic.top_n_leaves(top_n)

    @abstractmethod
    def _search(self, root: Node, nb_of_tree_walks: int):
        ...

