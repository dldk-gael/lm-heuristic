from tqdm.autonotebook import tqdm
from typing import *
import random
import numpy as np
from time import time
import pandas as pd
import seaborn as sns

from tree_search.strategy.search import TreeSearch
from tree_search.tree import Node


class RandomSearch(TreeSearch):
    """
    Perfom a random search in the tree
    """

    def __init__(
        self,
        root: Node,
        evaluation_fn: Callable[[List[Node]], List[float]],
        n_samples: int = 1,
        batch_size: int = 1,
    ):
        """
        :param root : Node from which the search will start
        :param evaluation_fn : function which give a score to terminal node
        :param n_samples: number of random expensions that will be computed
        :param batch_size: number of terminal nodes to store in a buffer before evaluating them in an single batch
        """
        TreeSearch.__init__(self, root, evaluation_fn)
        self.__path = []
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.__time = 0
        self.__values = []

    def random_expansion(self) -> (Node, List[Node]):
        """
        Perform a random expension from the root to a terminal node
        :return : the terminal node and the path taken
        """
        node = self.root
        path = [node]
        while True:
            if node.is_terminal():
                return node, path
            node = random.choice(node.childrens())
            path.append(node)

    def __best_in_buffer(self, buffer) -> (Node, List[Node], float):
        """
        :param buffer list of terminal nodes
        :return tuple(Node, float, List[Node]) the best terminal node contained in the buffer, its path and its value
        """
        scores = self._eval_node([n[0] for n in buffer])
        self.__values += scores
        return buffer[np.argmax(scores)] + (max(scores),)

    def search(self) -> Node:
        """
        Perform n_samples random expension and return the terminal node that has been found
        and that maximize the evaluation_fn
        """
        begin_time = time()

        best_node = None
        buffer = []
        best_node_value = -1
        for _ in tqdm(range(self.n_samples)):
            buffer.append(self.random_expansion())
            if len(buffer) == self.batch_size:
                best_buffer_node, path, best_buffer_node_value = self.__best_in_buffer(
                    buffer
                )
                if best_buffer_node_value > best_node_value:
                    best_node, self.__path, best_node_value = (
                        best_buffer_node,
                        path,
                        best_buffer_node_value,
                    )
                buffer = []

        if len(buffer) > 0:
            best_buffer_node, path, best_buffer_node_value = self.__best_in_buffer(
                buffer
            )
            if best_buffer_node_value > best_node_value:
                best_node, self.__path, best_node_value = (
                    best_buffer_node,
                    path,
                    best_buffer_node_value,
                )

        self.__time = time() - begin_time
        return best_node

    def path(self) -> List[Node]:
        """
        :return path taken from root node to best terminal node that has been found
        """
        assert self.__path != [], "Requesting best path but no search was performed yet"
        return self.__path

    def search_info(self):
        return {
            "time": self.__time,
            "path": self.__path,
            "total_nb_of_walks": self.n_samples,
            "best_leaf": self.__path[-1],
            "best_leaf_value": self._eval_node([self.__path[-1]])[0],
        }

    def plot_leaf_values_distribution(self):
        assert (
            self.__values != []
        ), "Try to plot leaf values distribution, but no search was performed yet"

        values = pd.Series(self.__values, name="Leaf values")
        sns.set()
        sns.distplot(values)
