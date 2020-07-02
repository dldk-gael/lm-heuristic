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

logging.basicConfig(level=logging.DEBUG)
random.seed(3)

GRAMMAR_FOLDER = "data/cfg/"
GRAMMAR_NAME = "ex_1_small"
BATCH_SIZE = 4

if __name__ == "__main__":
    # Load grammar tree
    logging.info("Load Grammar")
    grammar_root = CFGrammarNode.from_cfg_file(GRAMMAR_FOLDER + GRAMMAR_NAME + ".cfg", shrink=True)

    # Initialize the sentence scorer
    logging.info("Initialize the scorer")
    sentence_scorer = GPT2Score("gpt2", batch_size=BATCH_SIZE, length_normalization=True)
    # sentence_scorer = test_scorer
    # Ressource distributor
    ressource_distributor = RessourceDistributor(AllocationStrategy.ALL_FROM_ROOT)

    # Initialize the search parameters
    mcts = MonteCarloTreeSearch(
        sentence_scorer=sentence_scorer,
        buffer_size=BATCH_SIZE,
        ressource_distributor=ressource_distributor,
        nb_random_restarts=1,
        ucb_function=standart_ucb,
        parallel=True,
        progress_bar=True,
    )

    # Perform the search and print some info
    best_node, best_value = mcts(grammar_root, nb_of_tree_walks=4)
    print(str(best_node))
    mcts.shut_down()
