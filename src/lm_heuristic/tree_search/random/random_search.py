from typing import *
import random
import numpy as np

from lm_heuristic.tree_search import TreeSearch
from lm_heuristic.tree import Node
from lm_heuristic.heuristic import Heuristic


class RandomSearch(TreeSearch):
    """
    Perfom a random search in the tree
    """

    def __init__(self, heuristic: Heuristic, buffer_size: int = 1, verbose: bool = False):
        """
        :param heuristic : heuristic to eval leaves score
        :param buffer_size: number of terminal nodes to store in a buffer before evaluating them in an single batch
        """
        TreeSearch.__init__(self, heuristic, buffer_size)
        self._path: List[Node] = []
        self.verbose = verbose
        self._name = "Random Search"

        if self.verbose:
            print("--- INITIALIZATION ---\n %s\n" % str(self))

    @staticmethod
    def _random_expansion(root: Node) -> Tuple[Node, List[Node]]:
        """
        Perform a random expension from the root to a terminal node
        :return : the terminal node and the path taken
        """
        node = root
        path = [node]
        while True:
            if node.is_terminal():
                return node, path
            node = random.choice(node.childrens())
            path.append(node)

        return node, path

    def _best_in_buffer(self, buffer: List[Tuple[Node, List[Node]]]) -> Tuple[Node, List[Node], float]:
        """
        :param buffer list of terminal nodes
        :return tuple(Node, float, List[Node]) the best terminal node contained in the buffer, its path and its value
        """
        scores = self.heuristic.eval([n[0] for n in buffer])
        return buffer[np.argmax(scores)] + (max(scores),)

    def _search(self, root: Node, nb_of_tree_walks: int):
        """
        :param root : Node from which the search will start
        :param nb_of_tree_walks: number of random expensions that will be computed

        Perform nb_of_tree_walks random expension and return the terminal node that has been found
        and that maximize the evaluation_fn
        """
        if self.verbose:
            print("--- SEARCHING ---")

        buffer: List[Tuple[Node, List[Node]]] = []

        def flush_buffer():
            nonlocal buffer, self
            buffer_eval_results = self._best_in_buffer(buffer)
            best_leaf_in_buffer = buffer_eval_results[0]
            best_path_in_buffer = buffer_eval_results[1]
            best_leaf_value_in_buffer = buffer_eval_results[2]
            if best_leaf_value_in_buffer > self.best_leaf_value:
                self.best_leaf = best_leaf_in_buffer
                self.best_leaf_value = best_leaf_value_in_buffer
                self._path = best_path_in_buffer
            buffer = []

        for i in range(nb_of_tree_walks):
            if self.verbose and int(i / nb_of_tree_walks * 100) % 5 == 0:
                print("\rtree walks performed: %d%%" % int(i / nb_of_tree_walks * 100), end="")
            buffer.append(self._random_expansion(root))
            if len(buffer) == self.buffer_size:
                flush_buffer()

        if len(buffer) > 0:
            flush_buffer()

        if self.verbose:
            print("\rtree walks performed: 100%\n")

    def path(self) -> List[Node]:
        """
        :return path taken from root node to best terminal node that has been found
        """
        assert self._path != [], "Requesting best path but no search was performed yet"
        return self._path
