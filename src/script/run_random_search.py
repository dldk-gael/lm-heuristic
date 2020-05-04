from tree_search.strategy import RandomSearch
from tree_search.tree import Derivation
from heuristic import GPT2Score

import nltk
import matplotlib.pyplot as plt

"""
This script shows how to use a random searcher 
"""

GRAMMAR_FOLDER = 'data/cfg/'
if __name__ == '__main__':
    # Prepare root node
    root = Derivation.from_cfg_file(GRAMMAR_FOLDER+'ex_1.cfg')

    # Load heuristic function <- GPT2 score
    gpt_2_scorer = GPT2Score('gpt2', length_normalization=True, batch_size=16)
    heuristic = lambda terminal_nodes: gpt_2_scorer(list(map(str, terminal_nodes)))

    # Initialize and perform the search
    random_search = RandomSearch(root, evaluation_fn=heuristic, n_samples=100, batch_size=16)
    final_derivation = random_search.search()

    random_search.print_search_info()
    random_search.plot_leaf_values_distribution()
    plt.show()