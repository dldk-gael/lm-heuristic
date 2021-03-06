"""
Implementation of the Monte Carlo Tree Search (MCTS) algorithm
It follows mainly the ideas described in Single-Player Monte-Carlo Tree Search paper
from Maarten P.D. Schadd, Mark H.M. Winands, H. Jaap van den Herik,
Guillaume M.J-B. Chaslot, and Jos W.H.M. Uiterwijk
"""

from typing import *
import logging

from tqdm import tqdm

from lm_heuristic.tree import Node
from lm_heuristic.tree_search import TreeSearch, Evaluator
from lm_heuristic.utils.timer import time_function
from .evaluation_buffer import EvalBuffer, ParallelEvalBuffer
from .selection_rules import single_player_ucb
from .counter_node import CounterNode
from .ressource_distributor import RessourceDistributor, AllocationStrategy


logger = logging.getLogger(__name__)


class MonteCarloTreeSearch(TreeSearch):
    """
    This class will maintain a spanning tree (ST) over the nodes that have been visited by
    progressively wrapping the tree that need to be searched by a tree_search.CounterNode object

    At t=0, ST = {root}
    At time t:
        1. Selection phase: go to the frontier of ST using a specific selection policy
        2. Expansion phase: compute the children of the frontier node
        3. Simulation phase: choose a random leaf that is accessible from the frontier node
        4. Evaluation phase: evaluate the leaf 
        5. Backpropagation phrase: backpropagate the leaf value in the ST

    The evaluation task is not performed directly but is managed by an evaluation buffer (EvalBuffer object).
    -> The main motivation is to be able to input the leave by batch into the evaluation function.

    In this MCTS implementation, it is possible to:
    - progressively go down the tree rather that launching every tree walks from tree root
        -> the way to divide the computionnal ressources at each depth is specified by passing a RessourceDistributor
        -> there is two way to select the root's child :
            1. top_child -> select the child from which the best leaf have been found
            2. most_visited -> select the most visited child

    - make a certain number of random restarts to diminish the results' variance

    - choose different selection policy
        -> this policy is specified by passing a ucb function

    - choose to accumulate a certain amount of leaves before evaluating in one pass

    - perform the evaluation in another thread/process
    """

    def __init__(
        self,
        evaluator: Evaluator,
        buffer_size: int = 1,
        ressource_distributor: RessourceDistributor = None,
        expansion_threshold: int = 0,
        ucb_function: Callable[[CounterNode, CounterNode], float] = None,
        child_root_selection: str = "top_child",
        nb_random_restarts=1,
        name: str = "MCTS",
        progress_bar: bool = False,
        parallel_strategy: str = "none",
    ):
        TreeSearch.__init__(self, evaluator, name, progress_bar)
        self.expansion_threshold = expansion_threshold
        self.ressource_distributor = (
            ressource_distributor if ressource_distributor else RessourceDistributor(AllocationStrategy.ALL_FROM_ROOT)
        )
        self.nb_random_restarts = nb_random_restarts
        self.ucb_function = ucb_function if ucb_function else single_player_ucb

        if parallel_strategy == "none":
            self.eval_buffer = EvalBuffer(buffer_size, self._evaluator)
        else:
            self.eval_buffer = ParallelEvalBuffer(buffer_size, self._evaluator, parallel_strategy)

        self.child_root_selection = child_root_selection

    def _search(self, root: Node, nb_of_tree_walks: int):
        nb_tree_walks_per_search = nb_of_tree_walks // self.nb_random_restarts

        self.ressource_distributor.initialization(ressources=nb_tree_walks_per_search, tree=root)
        for i in range(self.nb_random_restarts):
            logger.info("Performing random restarts n°%d/%d", i + 1, self.nb_random_restarts)
            self.ressource_distributor.reset_remaining_ressources(nb_tree_walks_per_search)
            self._single_search(root)

    def _single_search(self, root: Node):
        counter_root = CounterNode(reference_node=root, parent=None)
        current_depth = 1
        self.ressource_distributor.set_new_position(current_depth, counter_root)

        with tqdm(total=self.ressource_distributor._ressources_to_consume, disable=not self._progress_bar) as p_bar:

            while (
                not counter_root.reference_node.is_terminal()
                and not counter_root.solved
                and self.ressource_distributor.still_has_ressources()
            ):

                self.ressource_distributor.consume_one_unit()

                # The 5 steps of MCTS: selection, expansion, simulation, evaluation, backpropagation
                frontier_counter_node = self.selection_phase(counter_root)
                self.expansion_phase(frontier_counter_node)
                random_leaf = self.simulation_phase(frontier_counter_node)
                self.evaluation_phase(frontier_counter_node, random_leaf)
                self.backpropagation_phase()

                # After each iteration, we query the ressource distributor to know if we should continue
                # to perform the tree walks from current root or if we should go down to the best children
                if self.ressource_distributor.go_to_children():
                    # Force the evaluation of the leave that still remain in the buffer
                    self.eval_buffer.force_eval()
                    self.backpropagation_phase()

                    # Freeze current_root to avoid modifying the counter in futur backprops
                    # Choose the best node among the  and continue the search from here
                    counter_root.freeze = True

                    if self.child_root_selection == "top_child":
                        counter_root = counter_root.top_child()
                    else:
                        counter_root = counter_root.most_visited_child()

                    logger.info("Current MCTS root : <%s>", str(counter_root.reference_node))

                    current_depth += 1
                    self.ressource_distributor.set_new_position(current_depth, counter_root)

                p_bar.update(1)

        self.eval_buffer.force_eval()

    @time_function
    def selection_phase(self, root: CounterNode) -> CounterNode:
        node = root
        node.count += 1
        while not node.is_terminal():
            node = self.selection_policy(node)
            node.count += 1

        return node

    def selection_policy(self, counter_node: CounterNode) -> CounterNode:
        # if a children of current node has not been visited yet: visit it
        for children in counter_node.children():
            if children.count == 0:
                return children

        # else select the node that maximise the UCB among the nodes that have not been solved yet
        unsolved_children = [children for children in counter_node.children() if not children.solved]
        return max(unsolved_children, key=lambda node: self.ucb_function(node, counter_node))

    @time_function
    def expansion_phase(self, counter_node: CounterNode):
        if counter_node.count < self.expansion_threshold:
            return

        if counter_node.reference_node.is_terminal():
            # backpropagate the information to the parent in order to avoid selecting this node in the futur
            counter_node.set_as_solved()
            return 

        counter_node.expand()  

    @time_function
    def simulation_phase(self, counter_node: CounterNode) -> Node:
        return counter_node.reference_node.random_walk()

    @time_function
    def evaluation_phase(self, counter_node: CounterNode, leaf: Node):
        self.eval_buffer.add(counter_node, leaf)

    @time_function
    def backpropagation_phase(self):
        results = self.eval_buffer.pop_results()
        for counter_node, leaf, reward in results:
            counter_node.backpropagate(reward, leaf)
