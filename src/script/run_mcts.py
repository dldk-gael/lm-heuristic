from tree_search.tree import Derivation
from tree_search.strategy import MonteCarloTreeSearch
from heuristic import GPT2Score
import random

"""
This script shows how to use a MCTS searcher
"""
random.seed(3)
GRAMMAR_FOLDER = "data/cfg/"
GRAMMAR_NAME = "ex_2"
BATCH_SIZE = 1

if __name__ == "__main__":
    # Load grammar tree
    grammar_root = Derivation.from_cfg_file(GRAMMAR_FOLDER + GRAMMAR_NAME + ".cfg", shrink=True)

    # Load heuristic function <- GPT2 score
    gpt_2_scorer = GPT2Score("gpt2", batch_size=BATCH_SIZE, length_normalization=True)
    heuristic = lambda terminal_nodes: gpt_2_scorer(list(map(str, terminal_nodes)))

    # Initialize the search parameters
    mtcs = MonteCarloTreeSearch(
        evaluation_fn=heuristic,
        batch_size=BATCH_SIZE,
        c=1,
        d=1000,
        t=0,
    )

    # Perform the search and print some info
    best_node = mtcs(grammar_root, nb_of_tree_walks=150)
    mtcs.print_search_info()

