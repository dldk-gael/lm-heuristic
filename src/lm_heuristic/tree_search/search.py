from abc import ABC, abstractmethod
from typing import *
import pandas as pd
import seaborn as sns

from lm_heuristic.tree import Node
from lm_heuristic.utils.memory import Memory
from lm_heuristic.utils.timer import Timer, timeit


class TreeSearch(ABC, Timer):
    """
    Abstract class that define a tree searcher
    The objective of a tree searcher is to find the leaf that maximise an evaluation function
    """

    def __init__(self, name: str = "", progress_bar: bool = False):
        Timer.__init__(self)
        self._name = name
        self._progress_bar = progress_bar
        self._memory = Memory()

    @timeit
    def __call__(self, root: Node, nb_of_tree_walks: int) -> Tuple[Node, float]:
        self._memory.reset()
        self.reset_timer()
        self._search(root, nb_of_tree_walks)
        return self._memory.best_in_memory()

    def plot_leaf_values_distribution(self):
        """
        Plot the leaf value distribution
        """
        values = self._memory.history_values()
        assert values != [], "Try to plot leaf values distribution, but no search was performed yet"

        series_values = pd.Series(values, name="Leaf values")
        sns.set()
        sns.distplot(series_values, label=str(self))

    def set_name(self, name: str):
        self._name = name

    def __str__(self) -> str:
        return self._name

    def top_n_leaves(self, top_n: int = 1) -> List[Tuple[Node, float]]:
        return self._memory.top_n_best(top_n)

    @abstractmethod
    def _search(self, root: Node, nb_of_tree_walks: int):
        ...

