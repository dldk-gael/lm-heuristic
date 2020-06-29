from typing import *
from tqdm import tqdm

from lm_heuristic.tree_search import TreeSearch
from lm_heuristic.tree import Node
from lm_heuristic.heuristic import Heuristic


class RandomSearch(TreeSearch):
    def __init__(self, heuristic: Heuristic, buffer_size: int = 1, name="Random Search"):
        TreeSearch.__init__(self, heuristic, name)
        self.buffer_size = buffer_size

    def _search(self, root: Node, nb_of_tree_walks: int):
        buffer = []

        for _ in tqdm(range(nb_of_tree_walks)):
            buffer.append(root.random_walk())
            if len(buffer) == self.buffer_size:
                self._heuristic.eval(buffer)
                buffer = []

        if len(buffer) > 0:
            self._heuristic.eval(buffer)

