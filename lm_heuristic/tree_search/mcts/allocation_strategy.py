from enum import Enum
import math


class AllocationStrategy(Enum):
    UNIFORM = 1
    LINEAR = 2

    def __str__(self):
        if self == AllocationStrategy.UNIFORM:
            return "uniform"
        if self == AllocationStrategy.LINEAR:
            return "linear"


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
        branching_factor: int,
        min_ressources_per_move: int = 1,
    ):
        """
        :param allocation_strategy: UNIFORM or LINEAR
            if UNIFORM, the total ressources available will be equally divised among all layers
            if LINEAR, more ressources will be avaible for first move and less for deeper move
                       with a linear repartition across layer
        :param total_ressources: total ressources available for the entire search
        :param depth: mean depth of the tree
        :param branching_factor: mean branching factor of the tree
        :param min_ressources_per_move: min ressources that need to be use at each layer
        """
        self.allocation_strategy = allocation_strategy
        self.total_ressources = total_ressources
        self.depth = depth
        self.min_ressources_per_move = min_ressources_per_move

        # if uniform strategy is chosen, total_ressources // max_depth will be allowed at each layers
        # however, once near depth_max, we won't need all the ressources
        # more precisly at depth d = depth_max - ln(total_ressources // max_depth) / branching_factor
        # total_ressources // max_depth will be sufficient to visit all leaves
        # so we will re-attribute the ressources of child to parents
        # this is not an exact calculus but it will allow to better respect the total_ressources
        ressource_per_layer = total_ressources // depth
        nb_of_layers_non_evaluated = math.log(ressource_per_layer) / branching_factor
        self.correction = int(ressource_per_layer / (depth - nb_of_layers_non_evaluated))

        # if linear strategy is chosen, the parameter a and b are computed such as :
        #   f(d) = a * d + b give the number of ressources allowed at depth d
        #   f(d_max) = min_ressources per move
        #   sum over d (= 1 .. d_max) of f(d) = total_ressources
        if allocation_strategy == AllocationStrategy.LINEAR:
            self.a = 2 / (1 - depth) * (total_ressources / depth - min_ressources_per_move)
            self.b = min_ressources_per_move - 2 / (1 - depth) * (total_ressources - depth * min_ressources_per_move)

    def uniform(self) -> int:
        return self.total_ressources // self.depth + self.correction

    def linear(self, current_depth: int) -> int:
        return int(max(self.a * current_depth + self.b, self.min_ressources_per_move))

    def __call__(self, current_depth: int) -> int:
        if self.allocation_strategy == AllocationStrategy.UNIFORM:
            return self.uniform()
        if self.allocation_strategy == AllocationStrategy.LINEAR:
            return self.linear(current_depth)
