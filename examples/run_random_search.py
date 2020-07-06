"""
This script shows how to use a random searcher and plot the distribution of leaf values
"""
import logging

import matplotlib.pyplot as plt

from lm_heuristic.tree_search import Evaluator
from lm_heuristic.tree_search.random import RandomSearch
from lm_heuristic.tree.interface.nltk_grammar import CFGrammarNode
from lm_heuristic.sentence_score import GPT2Score


GRAMMAR_FOLDER = "data/cfg/"
GRAMMAR_NAME = "ex_4"
logging.basicConfig(level=logging.INFO)
logging.getLogger('transformers').addHandler(logging.NullHandler())

if __name__ == "__main__":
    logging.info("Prepare root node")
    root = CFGrammarNode.from_cfg_file(GRAMMAR_FOLDER + GRAMMAR_NAME + ".cfg")

    logging.info("Load heuristic function <- GPT2 score")
    gpt_2_scorer = GPT2Score("gpt2", length_normalization=True, batch_size=1)
    evaluator = Evaluator(lambda terminal_nodes: gpt_2_scorer(list(map(str, terminal_nodes))))

    logging.info("Initialize and perform the search")
    random_search = RandomSearch(evaluator)
    final_derivation = random_search.search(root, nb_of_tree_walks=100)

    random_search.plot_leaf_values_distribution()
    plt.show()
