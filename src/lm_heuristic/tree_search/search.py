"""
Define an abstract class from which all tree search must inherate
"""

from abc import ABC, abstractmethod
from typing import *

import pandas as pd
import seaborn as sns

from lm_heuristic.sentence_score import SentenceScore
from lm_heuristic.tree import Node
from lm_heuristic.utils.memory import Memory


class TreeSearch(ABC):
    """
    The objective of a tree searcher is to find the leaf that maximise an evaluation function. 
    More particullary, in this project context, the evaluation function is a sentence scorer and 
    will evaluated the string representation of a given leaf.

    Given an maximum number of tree walks, a tree searcher simply browse the tree and store all 
    leaf, value that have been found in a memory object
    """

    def __init__(
        self, sentence_scorer: SentenceScore, buffer_size: int = 1, name: str = "", progress_bar: bool = False
    ):
        self._sentence_scorer = sentence_scorer
        self._buffer_size = buffer_size
        self._name = name
        self._progress_bar = progress_bar
        self._memory = Memory()

    def __call__(self, root: Node, nb_of_tree_walks: int) -> Tuple[Node, float]:
        self._memory.reset()
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

