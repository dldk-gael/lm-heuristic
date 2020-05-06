from src.tree import CFGrammarNode
from src.tree_search.mcts import MonteCarloTreeSearch, AllocationStrategy
from src.heuristic import Heuristic
from src.heuristic.sentence_score import GPT2Score
import random

"""
This script shows how to use a MCTS searcher
"""
random.seed(3)
GRAMMAR_FOLDER = "../data/cfg/"
GRAMMAR_NAME = "ex_2"
BATCH_SIZE = 1

if __name__ == "__main__":
    # Load grammar tree
    grammar_root = CFGrammarNode.from_cfg_file(GRAMMAR_FOLDER + GRAMMAR_NAME + ".cfg", shrink=True)

    # Load heuristic function <- GPT2 score
    gpt_2_scorer = GPT2Score("gpt2", batch_size=BATCH_SIZE, length_normalization=True)
    evaluation_fn = lambda terminal_nodes: gpt_2_scorer(list(map(str, terminal_nodes)))
    heuristic = Heuristic(evaluation_fn)

    # Initialize the search parameters
    mtcs = MonteCarloTreeSearch(
        heuristic=heuristic,
        batch_size=BATCH_SIZE,
        c=1,
        d=1000,
        t=0,
        verbose=True
    )

    # Perform the search and print some info
    best_node = mtcs(grammar_root, nb_of_tree_walks=150)
    mtcs.print_search_info()

