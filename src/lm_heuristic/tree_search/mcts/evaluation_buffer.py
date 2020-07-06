"""
Define evaluation buffers that can be used between the MTCS and the evaluation of the leaves.

The advantages of using such buffer are:
1. we can seen node to the evaluator object by batch. And, in this project, the evaluation function (transformers-based NN)
are more efficient when the input are sent by batches.

2. by cleary separating the search and the evaluation tasks, it is more easy to allow parallel implementation of those tasks
"""

from typing import *
import logging
import queue
import multiprocessing
import threading

from lm_heuristic.tree import Node
from lm_heuristic.tree_search import Evaluator
from .counter_node import CounterNode

logger = logging.getLogger(__name__)

######################################################################
## Vanilla implementation of the Evaluation buffer.
## -> No parallelization here
######################################################################


class EvalBuffer:
    """
    The evaluation buffer works as follows:
    1. use add methods to add new leaf that need to be evaluated
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
        self, buffer_size: int, evaluator: Evaluator, load_LM_in_memory: bool = True
    ):
        self._buffer_size = buffer_size
        self._index_table: Dict[Node, List[CounterNode]] = dict()
        self._evaluator = evaluator

        # load_LM_in_memory condition enable to not directly load the model in memory for class that inherate of EvalBuffer
        if load_LM_in_memory:
            self._evaluator.build()

        self._results: List[Tuple[CounterNode, Node, float]] = []

    def add(self, counter_node: CounterNode, leaf: Node):
        if self._evaluator.has_already_eval(leaf):
            reward = self._evaluator.value_from_memory(leaf)
            self._results.append((counter_node, leaf, reward))

        else:
            self._index_table.setdefault(leaf, [])
            self._index_table[leaf].append(counter_node)

            if len(self._index_table) == self._buffer_size:
                self._compute()

    def _compute(self):
        leaves = list(self._index_table.keys())
        results = self._evaluator.eval(leaves)
        self._handle_results(leaves, self._index_table, results)
        self._index_table = dict()

    def _handle_results(self, leaves, index_table, results):
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
        evaluator: Evaluator,
        tasks_queue: Union[queue.Queue, multiprocessing.Queue],
        results_queue: Union[queue.Queue, multiprocessing.Queue],
    ):
        self._evaluator = evaluator
        self._tasks_queue = tasks_queue
        self._results_queue = results_queue

    def __call__(self):
        logger.info("Evaluation worker correctly launched")
        self._evaluator.build()  # Load the language model in memory
        while True:
            leave_to_eval = self._tasks_queue.get(block=True)
            scores = self._evaluator.eval(leave_to_eval)
            self._results_queue.put(scores)


class MultithreadEvalWorker(ParallelEvalWorker, threading.Thread):
    """
    Same as ParallelEvalWorker but specific to multithread parallelisation
    """

    def __init__(self, sentence_scorer, tasks_queue, results_queue):
        threading.Thread.__init__(self)
        ParallelEvalWorker.__init__(self, sentence_scorer, tasks_queue, results_queue)

    def run(self):
        self()


class ParallelEvalBuffer(EvalBuffer):
    """
    Same as EvalBuffer except that the evaluation part is executed by an evaluation worker
    which is runned either on another thread (if parallel_strategy = multithread)
    or in another process (if parallel_strategy = multiprocess)
    """

    def __init__(
        self,
        buffer_size: int,
        evaluator: Evaluator,
        parallel_strategy: str = "multithread",
        max_nb_of_tasks_in_advance: int = 2,
    ):
        EvalBuffer.__init__(self, buffer_size, evaluator, load_LM_in_memory=False)
        self._tasks_queue: Union[queue.Queue, multiprocessing.Queue]
        self._results_queue: Union[queue.Queue, multiprocessing.Queue]
        self._eval_worker: Union[multiprocessing.Process, MultithreadEvalWorker]

        if parallel_strategy == "multithread":
            logger.info("Create a new thread to evaluate the leaves")
            self._tasks_queue = queue.Queue()
            self._results_queue = queue.Queue()
            self._eval_worker = MultithreadEvalWorker(evaluator, self._tasks_queue, self._results_queue)

        elif parallel_strategy == "multiprocess":
            logger.info("Create a new process to evaluate the leaves")
            self._tasks_queue = multiprocessing.Queue()
            self._results_queue = multiprocessing.Queue()
            self._eval_worker = multiprocessing.Process(
                target=ParallelEvalWorker(evaluator, self._tasks_queue, self._results_queue)
            )

        self._eval_worker.daemon = True
        self._eval_worker.start()
        self._in_progress_tasks = []  # Use to keep track of the tasks that was given to the worker
        self._max_nb_of_tasks_in_advance = max_nb_of_tasks_in_advance

    def _compute(self):
        leave = list(self._index_table.keys())
        if len(self._in_progress_tasks) == self._max_nb_of_tasks_in_advance:
            self._retrieve_from_results_queue(block=True)

        self._tasks_queue.put(leave)
        self._in_progress_tasks.append((leave, self._index_table))
        self._index_table = dict()

    def _retrieve_from_results_queue(self, block=False):
        results = self._results_queue.get(block=block)
        leave, index_table = self._in_progress_tasks.pop(0)
        self._handle_results(leave, index_table, results)

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
