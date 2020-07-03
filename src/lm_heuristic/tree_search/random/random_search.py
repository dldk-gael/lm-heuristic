"""
Implement a random searcher for baseline
"""

from typing import *
from tqdm import tqdm

from lm_heuristic.tree_search import TreeSearch
from lm_heuristic.tree import Node
from lm_heuristic.sentence_score import SentenceScore


class RandomSearch(TreeSearch):
    """
    Randomly sample the tree. 
    Except sending the leave by batches, there is no optimization at all.
    """
    def __init__(
        self,
        sentence_scorer: SentenceScore,
        buffer_size: int = 1,
        name: str = "Random Search",
        progress_bar: bool = False,
    ):
        TreeSearch.__init__(self, sentence_scorer, buffer_size, name, progress_bar)

    def _search(self, root: Node, nb_of_tree_walks: int):
        leave_buffer = []

        for _ in tqdm(range(nb_of_tree_walks), disable=not self._progress_bar):
            leave_buffer.append(root.random_walk())
            if len(leave_buffer) == self._buffer_size:
                scores = self._sentence_scorer.compute_score(list(map(str, leave_buffer)))
                self._memory.update_memory(zip(leave_buffer, scores))
                leave_buffer = []

        if len(leave_buffer) > 0:
            scores = self._sentence_scorer.compute_score(list(map(str, leave_buffer)))
            self._memory.update_memory(zip(leave_buffer, scores))
