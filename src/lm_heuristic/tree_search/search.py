"""
Define an abstract class from which all tree search must inherate
"""

from abc import ABC, abstractmethod
from typing import *

import pandas as pd
import seaborn as sns

from lm_heuristic.tree import Node
from lm_heuristic.utils.timer import time_function, Timer
from .evaluator import Evaluator

class TreeSearch(ABC, Timer):
    """
    Given a root (tree.Node) and an evaluation function (tree_search.Evaluator), the goal of
    a tree_search object is to find a leaf that can be reached from the root and that
    maximise the evaluation function.
    """

    def __init__(
        self, evaluator: Evaluator, name: str = "", progress_bar: bool = False
    ):
        Timer.__init__(self)
        self._evaluator = evaluator
        self._name = name
        self._progress_bar = progress_bar

    @time_function
    def search(self, root: Node, nb_of_tree_walks: int) -> Tuple[Node, float]:
        self._evaluator.reset()
        self.reset_timer()
        self._search(root, nb_of_tree_walks)
        return self._evaluator.best_result()

    def plot_leaf_values_distribution(self):
        values = self._evaluator.history_of_values()
        assert values != [], "Try to plot leaf values distribution, but no search was performed yet"

        series_values = pd.Series(values, name="Leaf values")
        sns.set()
        sns.distplot(series_values, label=str(self))

    def set_name(self, name: str):
        self._name = name

    def __str__(self) -> str:
        return self._name

    def top_n_leaves(self, top_n: int = 1) -> List[Tuple[Node, float]]:
        return self._evaluator.top_n_best(top_n)

    @abstractmethod
    def _search(self, root: Node, nb_of_tree_walks: int):
        ...

