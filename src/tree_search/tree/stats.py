from typing import *
import numpy as np
from tree_search.tree import Node


class TreeStats:
    """
    Class use to accumulate informations about a tree
    and compute several statistics on depth and branching factor
    """
    def __init__(self, root: Node):
        """
        :param root: root of the tree to evaluate
        """
        self.root = root
        self._depths = []
        self._branching_factors = dict()

    def accumulate_stats(self, nb_samples:int=1):
        """
        Accumulate statistics on the tree by performing nb_samples tree walks
        :param nb_samples: number of tree walks to perform
        """
        self._depths = []
        self._branching_factors = dict()
        for _ in range(nb_samples):
            self.single_tree_walk()

    def single_tree_walk(self):
        """
        Perform a single tree walk and update statistics value
        """
        node = self.root
        depth = 0
        while not node.is_terminal():
            branching_factor = len(node.childrens())
            self._branching_factors.setdefault(depth, []).append(branching_factor)
            node = node.random_children()
            depth += 1
        self._depths.append(depth)

    def depths_info(self) -> Dict:
        assert (
            self._depths != []
        ), "Try to access statistic informations before browsing the tree"
        return {
            "mean": np.mean(self._depths),
            "median": np.median(self._depths),
            "std": np.std(self._depths),
        }

    def branching_factors_info(self) -> Dict:
        assert (
                self._depths != []
        ), "Try to access statistic informations before browsing the tree"
        mean_by_depth_b_factors = list(map(np.mean, self._branching_factors.values()))
        return {
            "mean": np.mean(mean_by_depth_b_factors),
            "median": np.median(mean_by_depth_b_factors),
            "std": np.std(mean_by_depth_b_factors),
        }

