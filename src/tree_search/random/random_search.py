from tqdm.autonotebook import tqdm
from typing import *
import random
import numpy as np

from tree_search import TreeSearch
from tree import Node
from heuristic import Heuristic


class RandomSearch(TreeSearch):
    """
    Perfom a random search in the tree
    """

    def __init__(
        self, heuristic: Heuristic, batch_size: int = 1,
    ):
        """
        :param heuristic : heuristic to eval leaves score
        :param batch_size: number of terminal nodes to store in a buffer before evaluating them in an single batch
        """
        TreeSearch.__init__(self, heuristic)
        self._path = []
        self.batch_size = batch_size

    @staticmethod
    def _random_expansion(root: Node) -> (Node, List[Node]):
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

    def _best_in_buffer(self, buffer) -> (Node, List[Node], float):
        """
        :param buffer list of terminal nodes
        :return tuple(Node, float, List[Node]) the best terminal node contained in the buffer, its path and its value
        """
        scores = self.heuristic.eval([n[0] for n in buffer])
        return buffer[np.argmax(scores)] + (max(scores),)

    def _search(self, root: Node, nb_of_tree_walks: int) -> (Node, float):
        """
        :param root : Node from which the search will start
        :param nb_of_tree_walks: number of random expensions that will be computed

        Perform nb_of_tree_walks random expension and return the terminal node that has been found
        and that maximize the evaluation_fn
        """
        best_leaf = None
        best_leaf_value = -1
        buffer = []

        def flush_buffer():
            nonlocal buffer, best_leaf, best_leaf_value, self
            buffer_eval_results = self._best_in_buffer(buffer)
            best_leaf_in_buffer = buffer_eval_results[0]
            best_path_in_buffer = buffer_eval_results[1]
            best_leaf_value_in_buffer = buffer_eval_results[2]
            if best_leaf_value_in_buffer > best_leaf_value:
                best_leaf = best_leaf_in_buffer
                best_leaf_value = best_leaf_value_in_buffer
                self.__path = best_path_in_buffer
            buffer = []

        for _ in tqdm(range(nb_of_tree_walks)):
            buffer.append(self._random_expansion(root))
            if len(buffer) == self.batch_size:
                flush_buffer()

        if len(buffer) > 0:
            flush_buffer()

        return best_leaf, best_leaf_value

    def path(self) -> List[Node]:
        """
        :return path taken from root node to best terminal node that has been found
        """
        assert self._path != [], "Requesting best path but no search was performed yet"
        return self._path

    def __str__(self) -> str:
        return "RandomSearch"
