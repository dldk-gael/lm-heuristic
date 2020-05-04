from tqdm.autonotebook import tqdm
from typing import *
import random
import numpy as np
from time import time


from tree_search.strategy.search import TreeSearch
from tree_search.tree import Node


class RandomSearch(TreeSearch):
    """
    Perfom a random search in the tree
    """

    def __init__(
        self, evaluation_fn: Callable[[List[Node]], List[float]], batch_size: int = 1,
    ):
        """
        :param evaluation_fn : function which give a score to terminal node
        :param batch_size: number of terminal nodes to store in a buffer before evaluating them in an single batch
        """
        TreeSearch.__init__(self, evaluation_fn)
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
        scores = self._eval_node([n[0] for n in buffer])
        return buffer[np.argmax(scores)] + (max(scores),)

    def _search(self, root: Node, nb_of_tree_walks: int) -> Node:
        """
        :param root : Node from which the search will start
        :param nb_of_tree_walks: number of random expensions that will be computed

        Perform nb_of_tree_walks random expension and return the terminal node that has been found
        and that maximize the evaluation_fn
        """
        best_node = None
        buffer = []
        best_node_value = -1
        for _ in tqdm(range(nb_of_tree_walks)):
            buffer.append(self._random_expansion(root))
            if len(buffer) == self.batch_size:
                best_buffer_node, path, best_buffer_node_value = self._best_in_buffer(
                    buffer
                )
                if best_buffer_node_value > best_node_value:
                    best_node, self._path, best_node_value = (
                        best_buffer_node,
                        path,
                        best_buffer_node_value,
                    )
                buffer = []

        if len(buffer) > 0:
            best_buffer_node, path, best_buffer_node_value = self._best_in_buffer(
                buffer
            )
            if best_buffer_node_value > best_node_value:
                best_node, self._path, best_node_value = (
                    best_buffer_node,
                    path,
                    best_buffer_node_value,
                )

        return best_node

    def path(self) -> List[Node]:
        """
        :return path taken from root node to best terminal node that has been found
        """
        assert self._path != [], "Requesting best path but no search was performed yet"
        return self._path

    def search_info(self) -> Tuple[List[Node], Node, float]:
        return self._path, self._path[-1], self._eval_node([self._path[-1]])[0]

    def __str__(self) -> str:
        return "RandomSearch"
