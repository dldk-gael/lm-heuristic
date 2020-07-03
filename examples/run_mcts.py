"""
This script shows how to use a MCTS searcher
"""

import random
import logging

from lm_heuristic.tree import CFGrammarNode
from lm_heuristic.tree_search.mcts import (
    MonteCarloTreeSearch,
    AllocationStrategy,
    RessourceDistributor,
    single_player_ucb,
    standart_ucb,
)
from lm_heuristic.sentence_score import GPT2Score, test_scorer
from lm_heuristic.utils.timer import TimeComputation, print_timer

logging.basicConfig(level=logging.DEBUG)
random.seed(3)

GRAMMAR_FOLDER = "data/cfg/"
GRAMMAR_NAME = "ex_1_small"
BATCH_SIZE = 5

if __name__ == "__main__":
    # Load grammar tree
    logging.info("Load Grammar")
    grammar_root = CFGrammarNode.from_cfg_file(GRAMMAR_FOLDER + GRAMMAR_NAME + ".cfg", shrink=True)

    # Initialize the sentence scorer
    logging.info("Initialize the scorer")
    #sentence_scorer = GPT2Score("gpt2", batch_size=BATCH_SIZE, length_normalization=True)
    # sentence_scorer = test_scorer
    # Ressource distributor
    ressource_distributor = RessourceDistributor(AllocationStrategy.ALL_FROM_ROOT)
 
    # Initialize the search parameters
    mcts_parallel = MonteCarloTreeSearch(
        sentence_scorer=test_scorer,
        buffer_size=1,
        ressource_distributor=ressource_distributor,
        nb_random_restarts=1,
        ucb_function=standart_ucb,
        parallel_strategy="multiprocess",
        progress_bar=True,
    )

    mcts_non_parallel = MonteCarloTreeSearch(
        sentence_scorer=test_scorer,
        buffer_size=1,
        ressource_distributor=ressource_distributor,
        nb_random_restarts=1,
        ucb_function=standart_ucb,
        parallel_strategy="none",
        progress_bar=True,
    )
    with TimeComputation("MCTS Total"):
        best_node, best_value = mcts_parallel(grammar_root, nb_of_tree_walks=512)
        print_timer(mcts_parallel.selection_phase)
        print_timer(mcts_parallel.expansion_phase)
        print_timer(mcts_parallel.simulation_phase)
        print_timer(mcts_parallel.evaluation_phase)
        print_timer(mcts_parallel.backpropagation_phase)

    with TimeComputation("MCTS non parallel Total"):
        best_node, best_value = mcts_non_parallel(grammar_root, nb_of_tree_walks=512)
        print_timer(mcts_non_parallel.selection_phase)
        print_timer(mcts_non_parallel.expansion_phase)
        print_timer(mcts_non_parallel.simulation_phase)
        print_timer(mcts_non_parallel.evaluation_phase)
        print_timer(mcts_non_parallel.backpropagation_phase)