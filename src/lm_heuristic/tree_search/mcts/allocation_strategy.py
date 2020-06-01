from enum import Enum
import math


class AllocationStrategy(Enum):
    UNIFORM = 1
    LINEAR = 2
    ALL_FROM_ROOT = 3

    def __str__(self):
        if self == AllocationStrategy.UNIFORM:
            return "uniform"
        if self == AllocationStrategy.LINEAR:
            return "linear"
        if self == AllocationStrategy.ALL_FROM_ROOT:
            return "all from root"


class RessourceAllocation:
    """
    This class is use to allocate computational ressources per "move"
    as describe in Single-Player Monte-Carlo Tree Search for SameGame by Schadda, Winandsan, Taka, Uiterwijka
    and refined in Attacking SameGame using Monte-Carlo Tree Search by Klein

    Because we are in a single player mode MCTS version, the main idea is to iteratively choose
    a move (ie: go down one layer deeper in the tree) and re-launch tree walks from new root

    This class specify how much ressources we will spent at each tree layer.
    """

    def __init__(
        self,
        allocation_strategy: AllocationStrategy,
        total_ressources: int,
        depth: int,
    ):
        """
        :param allocation_strategy: UNIFORM, LINEAR, ALL_FROM_ROOT
            if UNIFORM, the total ressources available will be equally divised among all first 80% layers
            if LINEAR, more ressources will be avaible for first move and less for deeper move
                       with a linear repartition across layer
            if ALL_FROM_ROOT, all the ressources will be used from root without ever going deeper in the tree
        :param total_ressources: total ressources available for the entire search
        :param depth: mean depth of the tree
        :param branching_factor: mean branching factor of the tree
        :param min_ressources_per_move: min ressources that need to be use at each layer
        """
        self.allocation_strategy = allocation_strategy
        self.total_ressources = total_ressources
        self.depth = depth

        # if linear strategy is chosen, the parameter a and b are computed such as :
        #   f(d) = a * d + b give the number of ressources allowed at depth d
        #   f(mean_depth) = 0
        #   sum over d (= 1 .. mean_depth) of f(d) = total_ressources
        if allocation_strategy == AllocationStrategy.LINEAR:
            self.a = 2 / (1 - depth) * (total_ressources / (depth))
            self.b = - 2 / (1 - depth) * (total_ressources)

    def uniform(self) -> int:
        # if uniform strategy is chosen, total_ressources // max_depth will be allowed at each layers
        # however, once near depth_max, we won't need all the ressources
        # as a result, we arbitly divide the ressource among the first 80% first layer
        return max(round(self.total_ressources / (0.8 * self.depth)), 1)

    def linear(self, current_depth: int) -> int:
        # 80% to shift the ressource allocation toward the first depth
        # because near depth_max we do not need a lot of ressources
        return max(int(self.a * 0.8 * current_depth + self.b), 1)

    def all_from_root(self, current_depth: int) -> int:
        if current_depth == 1:
            return self.total_ressources
        else:
            raise ValueError(
                "Request nb tree walks allowed from depth != 1 but allocation strategy selected is all from root"
            )

    def __call__(self, current_depth: int) -> int:
        if self.allocation_strategy == AllocationStrategy.UNIFORM:
            return self.uniform()
        elif self.allocation_strategy == AllocationStrategy.LINEAR:
            return self.linear(current_depth)
        else:
            return self.all_from_root(current_depth)
