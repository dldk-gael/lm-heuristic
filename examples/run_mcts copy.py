"""
This script shows how to use a MCTS searcher with a nltk context free grammar
"""
from lm_heuristic.tree.interface.nltk_grammar import CFGrammarNode
from lm_heuristic.utils.zero_scorer import ZeroScorer
from lm_heuristic.tree_search import Evaluator
from lm_heuristic.tree_search.mcts import (
    MonteCarloTreeSearch,
    AllocationStrategy,
    RessourceDistributor,
    standart_ucb,
)
from lm_heuristic.utils.timer import print_timer
import logging


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

GRAMMAR_FOLDER = "data/cfg/"
GRAMMAR_NAME = "ex_1_large"
BATCH_SIZE = 5

if __name__ == "__main__":
    # Load grammar tree
    print("Load Grammar")
    grammar_root = CFGrammarNode.from_cfg_file(GRAMMAR_FOLDER + GRAMMAR_NAME + ".cfg")

    ressource_distributor = RessourceDistributor(AllocationStrategy.ALL_FROM_ROOT)

    scorer = ZeroScorer()

    # Initialize the search parameters
    mcts = MonteCarloTreeSearch(
        evaluator=Evaluator(scorer),
        buffer_size=1,
        ressource_distributor=ressource_distributor,
        nb_random_restarts=1,
        ucb_function=standart_ucb,
        parallel_strategy="none",
        progress_bar=True,
    )

    mcts.search(grammar_root, nb_of_tree_walks=10000)
    print(len(mcts._evaluator._call_history))