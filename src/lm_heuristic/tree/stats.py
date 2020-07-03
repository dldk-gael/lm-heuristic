from typing import *
import logging 

import numpy as np

from lm_heuristic.utils.timer import timeit, Timer
from .node import Node

logger = logging.getLogger(__name__)

class TreeStats(Timer):
    """
    Class use to accumulate informations about a tree
    and compute several statistics on depth and branching factor
    """

    def __init__(self, root: Node):
        """
        :param root: root of the tree to evaluate
        """
        Timer.__init__(self)
        self.root = root
        self._depths: List[int] = []
        self._branching_factors: Dict[int, List[int]] = dict()

    @timeit
    def accumulate_stats(self, nb_samples: int = 1):
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
        depth = 1  # by choice root's depth = 1 (and not 0)

        while not node.is_terminal():
            branching_factor = len(node.children())
            self._branching_factors.setdefault(depth, []).append(branching_factor)
            node = node.random_children()
            depth += 1
        self._depths.append(depth)

    @staticmethod
    def dict_info(array):
        return {
            "min": round(np.min(array), 1),
            "max": round(np.max(array), 1),
            "mean": round(float(np.mean(array)), 1),
            "median": round(float(np.median(array)), 1),
            "std": round(float(np.std(array)), 2),
        }

    def depths_info(self) -> Dict:
        assert self._depths != [], "Try to access statistic informations before browsing the tree"
        return self.dict_info(self._depths)

    def branching_factors_info(self) -> Dict:
        assert self._depths != [], "Try to access statistic informations before browsing the tree"
        mean_by_depth_b_factors = list(map(np.mean, self._branching_factors.values()))
        return self.dict_info(mean_by_depth_b_factors)
