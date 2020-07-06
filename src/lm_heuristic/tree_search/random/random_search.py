"""
Implement a random searcher for baseline
"""

from typing import *
from tqdm import tqdm

from lm_heuristic.tree_search import TreeSearch, Evaluator
from lm_heuristic.tree import Node


class RandomSearch(TreeSearch):
    """
    Randomly sample the tree. 
    Except sending the leave by batches, there is no optimization at all.
    """
    def __init__(
        self,
        evaluator: Evaluator,
        buffer_size: int = 1,
        name: str = "Random Search",
        progress_bar: bool = False,
    ):
        TreeSearch.__init__(self, evaluator, name, progress_bar)
        self._buffer_size = buffer_size

    def _search(self, root: Node, nb_of_tree_walks: int):
        leave_buffer = []

        for _ in tqdm(range(nb_of_tree_walks), disable=not self._progress_bar):
            leave_buffer.append(root.random_walk())
            if len(leave_buffer) == self._buffer_size:
                self._evaluator.eval(leave_buffer)
                leave_buffer = []

        if len(leave_buffer) > 0:
            self._evaluator.eval(leave_buffer)
