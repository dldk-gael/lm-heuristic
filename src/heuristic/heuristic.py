from typing import *

from tree import Node
from utils.timer import timeit, Timer


class Heuristic(Timer):
    """
    Heuristic class is used to encapsulate an evaluation fonction and add the following feature:
    - track time spent
    - add a memory to not re-evaluate a leaf
    """

    evaluation_fct: Callable[[List[Node]], List[float]]
    memory: Dict[int, float]
    history: List[Tuple[Node, float]]

    def __init__(self, evaluation_fct: Callable[[List[Node]], List[float]]):
        Timer.__init__(self)
        self.memory = dict()
        self.evaluation_fct = evaluation_fct
        self.history = []

    @timeit
    def has_already_eval(self, node: Node) -> bool:
        """
        return True if a leaf has already been evaluated
        """
        return str(node) in self.memory

    @timeit
    def value_from_memory(self, node: Node) -> float:
        self.history.append((node, self.memory[hash(node)]))
        return self.memory[hash(node)]

    @timeit
    def eval(self, nodes: List[Node]) -> List[float]:
        """
        evaluate the leaves and store them in memory
        """
        values = self.evaluation_fct(nodes)

        # update memory
        for node, value in zip(nodes, values):
            self.memory[hash(node)] = value

        # update history
        self.history += list(zip(nodes, values))

        return values

    def reset(self):
        self.reset_timer()
        self.history = []
        self.memory = dict()

    def history_of_terminal_nodes(self):
        return [x[0] for x in self.history]

    def history_of_values(self):
        return [x[1] for x in self.history]