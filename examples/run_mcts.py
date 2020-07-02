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
from lm_heuristic.heuristic import Heuristic
from lm_heuristic.sentence_score import GPT2Score

logging.basicConfig(level=logging.INFO)
random.seed(3)

GRAMMAR_FOLDER = "data/cfg/"
GRAMMAR_NAME = "ex_1_small"
BATCH_SIZE = 1

if __name__ == "__main__":
    # Load grammar tree
    grammar_root = CFGrammarNode.from_cfg_file(GRAMMAR_FOLDER + GRAMMAR_NAME + ".cfg", shrink=True)

    # Load heuristic function <- GPT2 score
    gpt_2_scorer = GPT2Score("gpt2", batch_size=BATCH_SIZE, length_normalization=True)
    evaluation_fn = lambda terminal_nodes: gpt_2_scorer(list(map(str, terminal_nodes)))
    heuristic = Heuristic(evaluation_fn, binarize_results=True)

    # Ressource distributor
    ressource_distributor = RessourceDistributor(AllocationStrategy.ALL_FROM_ROOT)

    # Initialize the search parameters
    mcts = MonteCarloTreeSearch(
        heuristic=heuristic,
        buffer_size=BATCH_SIZE,
        ressource_distributor=ressource_distributor,
        nb_random_restarts=2,
        ucb_function=standart_ucb,
    )

    # Perform the search and print some info
    best_node, best_value = mcts(grammar_root, nb_of_tree_walks=512)
    mcts.print_search_info()
