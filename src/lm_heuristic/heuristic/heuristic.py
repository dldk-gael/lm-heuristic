from typing import *

from lm_heuristic.tree import Node
from lm_heuristic.utils.timer import timeit, Timer


class Heuristic(Timer):
    """
    Heuristic class is used to encapsulate an evaluation fonction and add the following feature:
    - track time spent a evaluating the leaf
    - a memory to not re-evaluate a leaf twice
    - the possibilty to set default value for particular node
    - a counter the number of a leaf that have been send to the evaluation function
    - the possibility to binarize the results
        -> instead of returning the raw evaluation results it will return :
            1 if results > average of all results so far, 0 else
    """

    def __init__(self, evaluation_fct: Callable[[List[Node]], List[float]], binarize_results: bool = False):
        Timer.__init__(self)
        self._memory: Dict[Node, float] = dict()
        self._default_values: Dict[Node, float] = dict()
        self._evaluation_fct = evaluation_fct
        self._history: List[Tuple[Node, float]] = []
        self._eval_counter = 0
        self._binarize_results = binarize_results
        self._average = 0.0
        self._best_node: Node
        self._best_value = 0.0

    def set_default_value(self, default_values: Dict[Node, float]):
        # The default value are directly pushed to the memory buffer
        self._default_values = default_values
        self._memory.update(self._default_values)

    def has_already_eval(self, node: Node) -> bool:
        return node in self._memory

    def value_from_memory(self, node: Node) -> float:
        value = self._memory[node]
        output = value if not self._binarize_results else float(value > self._average)

        self._average += value / (len(self._history) + 1)
        self._history.append((node, value))

        return output

    @timeit
    def eval(self, nodes: List[Node]) -> List[float]:
        """
        evaluate the leaves and store them in memory
        note that it do not check that if the leaves have been evaluating before
        """
        values = self._evaluation_fct(nodes)

        outputs = []
        # update memory and average
        for node, value in zip(nodes, values):
            outputs.append(value if not self._binarize_results else float(value > self._average))
            if value > self._best_value:
                self._best_node = node
                self._best_value = value

            self._average += value / (len(self._history) + 1)
            self._memory[node] = value  

        # update history and counter
        self._history += list(zip(nodes, values))
        self._eval_counter += len(nodes)

        return outputs

    def best_node_evaluated(self) -> Tuple[Node, float]:
        return self._best_node, self._best_value

    def top_n_leaves(self, top_n: int = 1) -> List[Tuple[Node, float]]:
        """
        return the top n best leaves encounter during the search
        using the heuristic memory
        """
        unique_leaf_value = list(set(self._history))
        return sorted(unique_leaf_value, key=lambda x: x[1], reverse=True)[:top_n]

    def reset(self):
        self.reset_timer()
        self._history = []
        self._memory = self._default_values.copy()  # The default values are never reset
        self._average = 0.0
        self._best_node = None
        self._best_value = 0.0
        self._eval_counter = 0

    def history_of_terminal_nodes(self):
        return [x[0] for x in self._history]

    def history_of_values(self):
        return [x[1] for x in self._history]
