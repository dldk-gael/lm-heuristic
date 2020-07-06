"""
Define an evaluator object which takes as input a single terminal node (or a batch of node)
and return its value (or a batch of values)
"""

from typing import *

from lm_heuristic.tree import Node


class Evaluator:
    """
    The evaluator is used to wrap an evaluation function and add on top of it several features :
    - a memory: so that two leave representing the same value will never be input to the
    evaluation function twice.
    - a history: keep track of all the call that are make to the object
    """

    def __init__(self, evaluation_fct: Callable[[List[str]], List[float]]):
        self._evaluation_fct = evaluation_fct
        self._memory: Dict[Node, float] = dict()
        self._default_values: Dict[Node, float] = dict()
        self._call_history: List[Tuple[Node, float]] = list()
        self._best_node: Node
        self._best_value: float = -1.0

    def reset(self):
        self._memory = self._default_values.copy()
        self._call_history = list()
        self._best_node = None
        self._best_value = -1.0

    def build(self):
        self._evaluation_fct.build() # To load the LM in memory from the evaluator

    def set_default_value(self, node: Node, value: float):
        # Sometimes it is interesting to specificy default values
        # For instance, when using FeatureGrammarNode, we can set a value to DEAD_END node
        self._default_values[node] = value
        self._memory.update(self._default_values)

    def has_already_eval(self, node: Node) -> bool:
        return node in self._memory

    def value_from_memory(self, node: Node) -> float:
        self._call_history.append((node, self._memory[node]))
        return self._memory[node]

    def eval(self, nodes: List[Node]) -> List[float]:
        values = self._evaluation_fct(list(map(str, nodes)))

        for node, value in zip(nodes, values):
            self._memory[node] = value
            if value > self._best_value:
                self._best_node, self._best_value = node, value 

        self._call_history += list(zip(nodes, values))

        return values

    def history_of_terminal_nodes(self):
        return [x[0] for x in self._call_history]

    def history_of_values(self):
        return [x[1] for x in self._call_history]

    def top_n_best(self, top_n):
        return sorted(self._memory.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def best_result(self):
        return self._best_node, self._best_value