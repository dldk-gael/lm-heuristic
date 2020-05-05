from enum import Enum


class AllocationStrategy(Enum):
    UNIFORM = 1
    LINEAR = 2


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
        max_depth: int,
        min_ressources_per_move: int = 1,
    ):
        """
        :param allocation_strategy: UNIFORM or LINEAR
            if UNIFORM, the total ressources available will be equally divised among all layers
            if LINEAR, more ressources will be avaible for first move and less for deeper move
                       with a linear repartition across layer
        :param total_ressources: total ressources available for the entire search
        :param max_depth: max_depth of the tree
        :param min_ressources_per_move: min ressources that need to be use at each layer
        """
        self.allocation_strategy = allocation_strategy
        self.total_ressources = total_ressources
        self.max_depth = max_depth
        self.min_ressources_per_move = min_ressources_per_move

        # if linear strategy is chosen, the parameter a and b are computed such as :
        #   f(d) = a * d + b give the number of ressources allowed at depth d
        #   f(d_max) = min_ressources per move
        #   sum over d (= 1 .. d_max) of f(d) = total_ressources
        if allocation_strategy == AllocationStrategy.LINEAR:
            self.a = (
                2
                / (1 - max_depth)
                * (total_ressources / max_depth - min_ressources_per_move)
            )
            self.b = min_ressources_per_move - 2 / (1 - max_depth) * (
                total_ressources - max_depth * min_ressources_per_move
            )

    def uniform(self) -> int:
        return self.total_ressources // self.max_depth

    def linear(self, current_depth: int) -> int:
        """
        the parameter a and b are computed such as :
        -> f(d) = a * d + b give the number of ressources allowed at depth d
        -> f(d_max) = min_ressources per move
        -> sum over d (= 1 .. d_max) of f(d) = total_ressources
        """
        return int(max(self.a * current_depth + self.b, self.min_ressources_per_move))

    def __call__(self, current_depth: int) -> int:
        if self.allocation_strategy == AllocationStrategy.UNIFORM:
            return self.uniform()
        if self.allocation_strategy == AllocationStrategy.LINEAR:
            return self.linear(current_depth)
