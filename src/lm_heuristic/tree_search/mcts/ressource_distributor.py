"""
In a single-player setting, there are several possibilities to divide the computationnal ressource available
during the MCTS. We can either compute all the tree walks from the tree root or we can choose,
after having computed a certain amount of tree walks, to select and go to the best root's child and continue
the search from this new node. This allows to progressively narrow the search in the most promising part of tree.
This module implements different stategies to tackle this task.
"""

from typing import *
import logging
from enum import Enum

from lm_heuristic.tree import Node
from lm_heuristic.tree.stats import TreeStats
from .counter_node import CounterNode


logger = logging.getLogger(__name__)


class AllocationStrategy(Enum):
    """
    Use to select the different allocation strategy for computationnal ressources
    ALL_FROM_ROOT: compute all the tree walks from the tree root
    UNIFORM: the same amount of tree walks will be performed from each depth
    LINEAR: the amount of tree walks will be lineary dependant of the depth such that
            at depth max all ressources are consumed

    DYNAMIC: we only go to a child when there is a sufficient amount of difference between children expected reward
    """

    UNIFORM = 1
    LINEAR = 2
    ALL_FROM_ROOT = 3
    DYNAMIC = 4

    def __str__(self):
        if self == AllocationStrategy.UNIFORM:
            return "uniform"
        if self == AllocationStrategy.LINEAR:
            return "linear"
        if self == AllocationStrategy.ALL_FROM_ROOT:
            return "all from root"
        if self == AllocationStrategy.DYNAMIC:
            return "dynamic"


class RessourceDistributor:
    """
    A ressource distributor is first initialized with a total number of tree walks.
    For Uniform and Linear strategy, it is needed to know the average tree's depth in order 
    to correctly split the ressources. To do so, a tree stats sampling of the tree is performed.

    Durant the search : 
    - before each tree walk, the MCTS indicates to the ressource distributor that one tree walks will
    be now consummed.
    - after each tree walk, the ressource distributor to the MCTS:
        -> if we must to change the current root and go to a child (go_to_children method)
        -> if the search must be stopped because all the ressources were consummed (still_has_ressources method)
    """

    def __init__(self, strategy: AllocationStrategy, stats_samples=None, dynamic_ratio=None):
        self.strategy = strategy
        self.a: float
        self.b: float
        self.depth_max: int
        self._ressources_to_consume: int

        self._ressources_already_consumed: int
        self._ressources_to_consume_at_current_depth: int
        self._ressources_already_consumed_at_current_depth: int
        self._current_depth: int
        self._current_node: CounterNode

        if self.strategy == AllocationStrategy.DYNAMIC:
            assert dynamic_ratio is not None, "you must provide dynamic_ratio when using dynamic strategy"
            self.dynamic_ratio = dynamic_ratio

        if self.strategy == AllocationStrategy.UNIFORM or self.strategy == AllocationStrategy.LINEAR:
            assert stats_samples is not None
            self.stats_samples = stats_samples

    def initialization(self, tree: Node, ressources: int):
        self._ressources_to_consume = ressources
        self._ressources_already_consumed = 0

        logger.info("Initialize the ressource distributor for %s statregy", str(self.strategy))
        if self.strategy == AllocationStrategy.UNIFORM or self.strategy == AllocationStrategy.LINEAR:
            logger.info("Computing statistic values on the tree.")
            stats = TreeStats(tree)
            stats.accumulate_stats(nb_samples=self.stats_samples)
            self.depth_max = int(stats.depths_info()["mean"])
            logger.info(
                "Stats results :\n- depth = %s,\n- branching_factor = %s",
                str(stats.depths_info()),
                str(stats.branching_factors_info()),
            )

        if self.strategy == AllocationStrategy.LINEAR:
            self.a = 2 / (1 - self.depth_max) * (ressources / (self.depth_max))
            self.b = -2 / (1 - self.depth_max) * (ressources)

    def still_has_ressources(self) -> bool:
        return self._ressources_already_consumed < self._ressources_to_consume

    def consume_one_unit(self):
        self._ressources_already_consumed += 1
        self._ressources_already_consumed_at_current_depth += 1

    def go_to_children(self):
        # By default if all the ressources are consummed,
        # it indicates that it should not change current root position
        if not self.still_has_ressources():
            return False
            
        if self.strategy == AllocationStrategy.DYNAMIC:
            # TODO EXPAND THIS TO EVERY STRATEGY
            # OTHER POSSIBILITY, DIRECTLY HANDLE THIS CASE FROM GRAMMAR
            # Special case, not need to stay in a position where there is only one children
            children = self._current_node.children()
            if len(children) == 1:
                return True
            nb_of_wins = [child.sum_rewards for child in children]
            sorted_nb_of_wins = sorted(nb_of_wins, reverse=True)
            return sorted_nb_of_wins[0] >= self.dynamic_ratio * (sorted_nb_of_wins[1] + 1)

        return self._ressources_already_consumed_at_current_depth == self._ressources_to_consume_at_current_depth

    def reset_remaining_ressources(self, new_ressources):
        self._ressources_to_consume = new_ressources
        self._ressources_already_consumed = 0

    def set_new_position(self, new_depth, new_node):
        self._current_depth = new_depth
        self._current_node = new_node
        self._ressources_already_consumed_at_current_depth = 0

        if self.strategy == AllocationStrategy.UNIFORM:
            # if uniform strategy is chosen, total_ressources // max_depth will be allowed at each layers
            # however, once near depth_max, we won't need all the ressources
            # as a result, we arbitly divide the ressource among the first 80% first layer
            self._ressources_to_consume_at_current_depth = max(
                round(self._ressources_to_consume / (0.8 * self.depth_max)), 1
            )

        if self.strategy == AllocationStrategy.LINEAR:
            self._ressources_to_consume_at_current_depth = max(int(self.a * 0.8 * self._current_depth + self.b), 1)

        if self.strategy == AllocationStrategy.ALL_FROM_ROOT:
            self._ressources_to_consume_at_current_depth = self._ressources_to_consume

        if self.strategy == AllocationStrategy.DYNAMIC:
            logger.info("Current depth = %d. The nb of tree walks will be dynamically computed.", self._current_depth)
        else:
            logger.info(
                "Current depth = %d. %d tree walks will be performed",
                self._current_depth,
                self._ressources_to_consume_at_current_depth,
            )

