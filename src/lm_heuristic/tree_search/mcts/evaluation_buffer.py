"""
Define evaluation buffers that can be used between the Monte Carlo search and the evaluation of the leaves.

The advantage of using such buffer are:
1. the evaluation function used in this project (LM basedsentence scorer) are more efficient when the input
are sent by batches.

2. cleary separating the search and the evaluation allows parallel implementation of those tasks
"""

from typing import *
import logging
import queue
import multiprocessing
import threading

from lm_heuristic.tree import Node
from lm_heuristic.utils.memory import Memory
from lm_heuristic.sentence_score import SentenceScore
from .counter_node import CounterNode

logger = logging.getLogger(__name__)

######################################################################
## Vanilla implementation of the Evaluation buffer.
## -> No parallelization here
######################################################################


class EvalBuffer:
    """
    The evaluation buffer works as follows:
    1. use add methods to add new leaf that need to be evaluate
    2. use pop_results to retrieve the results if there are available

    Behind the scenes, it uses the following optimization:
    - each time a new leaf is add to the buffer, it checks if the leaf has not yet. If it is the case
    the buffer retrieves the value and put it directly in the output queue

    - if at any time the buffer contains couple of (counter_node, leaf) that share the same leaf
        ie: (counter_node_1, leaf_A), (counter_node_2, leaf_A)
    only one instance of the leaf will be sent to the evaluation function

    - the leaf are sent to the evaluation function by batch of buffer_size
    """

    def __init__(
        self, buffer_size: int, memory: Memory, sentence_scorer: SentenceScore, load_LM_in_memory: bool = True
    ):
        self._buffer_size = buffer_size
        self._index_table: Dict[Node, List[CounterNode]] = dict()
        self._memory = memory

        self._sentence_scorer = sentence_scorer
        # load_LM_in_memory condition enable to not directly load the model in memory for class that inherate of EvalBuffer
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


######################################################################
## Concurrent implementation of the Evaluation buffer
######################################################################


class ParallelEvalWorker:
    """
    Until it is killed, an evaluation worker constinously :
    1. wait for a batch of sentences coming from a tasks queue
    2. score the sentences using a LM-based sentence scorer
    3. put the results in a results queue
    """

    def __init__(
        self,
        sentence_scorer: SentenceScore,
        tasks_queue: Union[queue.Queue, multiprocessing.Queue],
        results_queue: Union[queue.Queue, multiprocessing.Queue],
    ):
        self._sentence_scorer = sentence_scorer
        self._tasks_queue = tasks_queue
        self._results_queue = results_queue

    def __call__(self):
        logger.info("Evaluation worker correctly launched")
        self._sentence_scorer.build()  # Load the language model in memory
        while True:
            sentences = self._tasks_queue.get(block=True)
            scores = self._sentence_scorer.compute_score(sentences)
            self._results_queue.put(scores)


class MultithreadEvalWorker(ParallelEvalWorker, threading.Thread):
    """
    Same that ParallelEvalWorker but specific to multithread parallelisation
    """

    def __init__(self, sentence_scorer, tasks_queue, results_queue):
        threading.Thread.__init__(self)
        ParallelEvalWorker.__init__(self, sentence_scorer, tasks_queue, results_queue)

    def run(self):
        self()


class ParallelEvalBuffer(EvalBuffer):
    """
    Same as EvalBuffer except that the evaluation part is executed by an evaluation work
    which is runned either on another thread (if parallel_strategy = multithread)
    or in another process (if parallel_strategy = multiprocess)
    """

    def __init__(
        self,
        buffer_size: int,
        memory: Memory,
        sentence_scorer: SentenceScore,
        parallel_strategy: str = "multithread",
        max_nb_of_tasks_in_advance: int = 2,
    ):
        EvalBuffer.__init__(self, buffer_size, memory, sentence_scorer, load_LM_in_memory=False)
        self._tasks_queue: Union[queue.Queue, multiprocessing.Queue]
        self._results_queue: Union[queue.Queue, multiprocessing.Queue]
        self._eval_worker: Union[multiprocessing.Process, MultithreadEvalWorker]

        if parallel_strategy == "multithread":
            logger.info("Create a new thread to evaluate the leaves")
            self._tasks_queue = queue.Queue()
            self._results_queue = queue.Queue()
            self._eval_worker = MultithreadEvalWorker(sentence_scorer, self._tasks_queue, self._results_queue)

        elif parallel_strategy == "multiprocess":
            logger.info("Create a new process to evaluate the leaves")
            self._tasks_queue = multiprocessing.Queue()
            self._results_queue = multiprocessing.Queue()
            self._eval_worker = multiprocessing.Process(
                target=ParallelEvalWorker(sentence_scorer, self._tasks_queue, self._results_queue)
            )

        self._eval_worker.daemon = True
        self._eval_worker.start()
        self._in_progress_tasks = []  # Use to keep track tasks given to the worker
        self._max_nb_of_tasks_in_advance = max_nb_of_tasks_in_advance

    def _compute(self):
        leaves = list(self._index_table.keys())
        sentences = list(map(str, leaves))
        if len(self._in_progress_tasks) == self._max_nb_of_tasks_in_advance:
            self._retrieve_from_results_queue(block=True)

        self._tasks_queue.put(sentences)
        self._in_progress_tasks.append((leaves, self._index_table))
        self._index_table = dict()

    def _retrieve_from_results_queue(self, block=False):
        results = self._results_queue.get(block=block)
        leaves, index_table = self._in_progress_tasks.pop(0)
        self._handle_results(leaves, index_table, results)

    def pop_results(self):
        while not self._results_queue.empty():
            self._retrieve_from_results_queue(block=False)

        results = self._results
        self._results = []
        return results

    def force_eval(self):
        if len(self._index_table) > 0:
            self._compute()

        while len(self._in_progress_tasks) > 0:
            self._retrieve_from_results_queue(block=True)
