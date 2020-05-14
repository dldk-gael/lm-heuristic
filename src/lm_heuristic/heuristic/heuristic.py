from typing import *

from lm_heuristic.tree import Node
from lm_heuristic.utils.timer import timeit, Timer


class Heuristic(Timer):
    """
    Heuristic class is used to encapsulate an evaluation fonction and add the following feature:
    - track time spent
    - add a memory to not re-evaluate a leaf
    """

    def __init__(
        self, evaluation_fct: Callable[[List[Node]], List[float]], use_memory: bool = False,
    ):
        Timer.__init__(self)
        self.memory: Dict[int, float] = dict()
        self.evaluation_fct = evaluation_fct
        self.history: List[Tuple[Node, float]] = []
        self.use_memory = use_memory
        self.eval_counter = 0

    @timeit
    def has_already_eval(self, node: Node) -> bool:
        """
        return True if a leaf has already been evaluated
        """
        return (str(node) in self.memory) if self.use_memory else False

    @timeit
    def value_from_memory(self, node: Node) -> float:
        assert self.use_memory, "Try to access memory, but use_memory is set to false"
        self.history.append((node, self.memory[hash(node)]))
        return self.memory[hash(node)]

    @timeit
    def eval(self, nodes: List[Node]) -> List[float]:
        """
        evaluate the leaves and store them in memory
        """
        values = self.evaluation_fct(nodes)

        if self.use_memory:
            # update memory
            for node, value in zip(nodes, values):
                self.memory[hash(node)] = value

        # update history and counter
        self.history += list(zip(nodes, values))
        self.eval_counter += len(nodes)

        return values

    def reset(self):
        self.reset_timer()
        self.history = []
        self.memory = dict()
        self.eval_counter = 0

    def history_of_terminal_nodes(self):
        return [x[0] for x in self.history]

    def history_of_values(self):
        return [x[1] for x in self.history]
