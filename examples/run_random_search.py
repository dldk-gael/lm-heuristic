"""
This script shows how to use a random searcher and plot the distribution of leaf values
"""

import matplotlib.pyplot as plt

from lm_heuristic.tree_search import Evaluator
from lm_heuristic.tree_search.random import RandomSearch
from lm_heuristic.tree.interface.nltk_grammar import CFGrammarNode
from lm_heuristic.sentence_score import GPT2Score


GRAMMAR_FOLDER = "data/cfg/"
GRAMMAR_NAME = "ex_4"

if __name__ == "__main__":
    print("Prepare root node")
    root = CFGrammarNode.from_cfg_file(GRAMMAR_FOLDER + GRAMMAR_NAME + ".cfg")

    print("Load heuristic function <- GPT2 score")
    gpt_2_scorer = GPT2Score("gpt2", batch_size=1, normalization_strategy="mean_length_log_prob").build()
    evaluator = Evaluator(gpt_2_scorer)

    print("Initialize and perform the search")
    random_search = RandomSearch(evaluator, progress_bar=True)
    final_derivation = random_search.search(root, nb_of_tree_walks=100)

    random_search.plot_leaf_values_distribution()
    plt.show()
