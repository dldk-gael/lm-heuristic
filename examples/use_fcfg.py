from lm_heuristic.tree import FeatureGrammarNode
from lm_heuristic.heuristic import Heuristic
from lm_heuristic.tree_search import RandomSearch

import random 

random.seed(4)

if __name__ == "__main__":
    root = FeatureGrammarNode.from_cfg_file("data/feat0.fcfg")

    evaluation_fct = lambda terminal_nodes: [0] * len(terminal_nodes)
    heuristic = Heuristic(evaluation_fct)
    searcher = RandomSearch(heuristic)

    leaf, _ = searcher(root, nb_of_tree_walks=1)
    print(leaf)