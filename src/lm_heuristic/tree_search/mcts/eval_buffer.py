from typing import *
from multiprocessing import Queue, Process

from lm_heuristic.tree import Node
from lm_heuristic.utils.memory import Memory
from lm_heuristic.sentence_score import SentenceScore
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

    def __init__(self, buffer_size: int, memory: Memory, sentence_scorer: SentenceScore, load_LM_in_memory=True):
        self._buffer_size = buffer_size
        self._index_table: Dict[Node, List[CounterNode]] = dict()
        self._memory = memory

        self._sentence_scorer = sentence_scorer
        if load_LM_in_memory:
            self._sentence_scorer.build()

        self._results: List[Tuple[CounterNode, Node, float]] = []

    def add(self, counter_node: CounterNode, leaf: Node):
        if self._memory.has_already_eval(leaf):
            reward = self._memory.value_from_memory(leaf)
            self._results.append((counter_node, leaf, reward))

        else:
            self._index_table.setdefault(leaf, [])
            self._index_table[leaf].append(counter_node)

            if len(self._index_table) == self._buffer_size:
                self._compute()

    def _compute(self):
        leaves = list(self._index_table.keys())
        sentences = list(map(str, leaves))
        results = self._sentence_scorer(sentences)
        self._handle_results(leaves, self._index_table, results)
        self._index_table = dict()

    def _handle_results(self, leaves, index_table, results):
        self._memory.update_memory(zip(leaves, results))
        for leaf, reward in zip(leaves, results):
            for counter_node in index_table[leaf]:
                self._results.append((counter_node, leaf, reward))

    def pop_results(self):
        results = self._results
        self._results = []
        return results

    def force_eval(self):
        if len(self._index_table) > 0:
            self._compute()


class ParallelEvalBuffer(EvalBuffer):
    def __init__(self, buffer_size: int, memory: Memory, sentence_scorer: SentenceScore):
        EvalBuffer.__init__(self, buffer_size, memory, sentence_scorer, load_LM_in_memory=False)
        self._tasks_queue: Queue = Queue()
        self._results_queue: Queue = Queue()
        self._worker = Process(
            target=self.evaluation_job, args=(self._tasks_queue, self._results_queue, sentence_scorer)
        )
        self._worker.start()
        self._in_progress_tasks = []

    @staticmethod
    def evaluation_job(input_queue: Queue, ouput_queue: Queue, sentence_scorer: SentenceScore):
        sentence_scorer.build()  # Load the language model in memory
        while True:
            sentences = input_queue.get(block=True)
            scores = sentence_scorer.compute_score(sentences)
            ouput_queue.put(scores)

    def _compute(self):
        leaves = list(self._index_table.keys())
        sentences = list(map(str, leaves))
        self._tasks_queue.put(sentences)
        self._in_progress_tasks.append((leaves, self._index_table))
        self._index_table = dict()

    def pop_results(self):
        while not self._results_queue.empty():
            results = self._results_queue.get(block=False)
            leaves, index_table = self._in_progress_tasks.pop(0)
            self._handle_results(leaves, index_table, results)

        results = self._results
        self._results = []
        return results

    def force_eval(self):
        if len(self._index_table) > 0:
            self._compute()

        while len(self._in_progress_tasks) > 0:
            print(self._in_progress_tasks[0])
            results = self._results_queue.get(block=True)
            leaves, index_table = self._in_progress_tasks.pop(0)
            self._handle_results(leaves, index_table, results)

    def kill_sub_process(self):
        self._worker.kill()
