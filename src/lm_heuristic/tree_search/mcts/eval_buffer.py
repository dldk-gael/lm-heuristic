from typing import *

from lm_heuristic.tree import Node
from lm_heuristic.heuristic import Heuristic

from .counter_node import CounterNode
 

class EvalBuffer:
    """
    Define a specific evaluation buffer for the MCTS algorithm
    that handle both heuristic evaluation of the leaves and backpropagation of the leaves' value

    Each time that a new couple (counter node, leaf) is added to the buffer,
    the buffer check if the leaf has not yet been evaluated before by the heuristic
        if it is the case, it retrieves the value and backpropagates it directly from the counter node
        else it is added in the buffer table.

    When the length of the buffer table reaches the buffer max size, all the leaves are evaluated
    and the values backpropagated from the counter node

    The buffer table has been designed to efficiently handle the specific case where several couple
    share the same leaf (counter_node_1, leaf_A), (counter_node_2, leaf_A), ...
    if such case occurs the leaf_A will only be evaluated once
    """

    def __init__(self, buffer_size: int, heuristic: Heuristic):
        self._buffer_size = buffer_size
        self._buffer_table: Dict[Node, List[CounterNode]] = dict()
        self._heuristic = heuristic

    def reset(self):
        self._buffer_table = dict()

    def add(self, counter_node: CounterNode, leaf: Node):
        if self._heuristic.has_already_eval(leaf):
            leaf_value = self._heuristic.value_from_memory(leaf)
            counter_node.backpropagate(leaf_value, leaf)

        else:
            self._buffer_table.setdefault(leaf, [])
            self._buffer_table[leaf].append(counter_node)

            if len(self._buffer_table) == self._buffer_size:
                self._compute()

    def force_eval(self):
        if len(self._buffer_table) > 0:
            self._compute()

    def _compute(self):
        leaves = list(self._buffer_table.keys())
        rewards = self._heuristic.eval(leaves)

        for leaf, reward in zip(leaves, rewards):
            for counter_node in self._buffer_table[leaf]:
                counter_node.backpropagate(reward, leaf)

        self.reset()
