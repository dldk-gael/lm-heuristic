"""
Use to evaluate depth and branching factor of tree by performing random sampling on it
"""

from lm_heuristic.tree.interface.nltk_grammar import CFGrammarNode
from lm_heuristic.tree.stats import TreeStats


GRAMMAR_FOLDER = "data/cfg/"
GRAMMAR_NAME = "ex_1_small"
NB_SAMPLES = 1000

if __name__ == "__main__":

    # Prepare grammar tree
    grammar_root = CFGrammarNode.from_cfg_file(GRAMMAR_FOLDER + GRAMMAR_NAME + '.cfg', shrink=True)
    print(grammar_root.children()[0])

    # Compute stats
    stats = TreeStats(grammar_root)
    stats.accumulate_stats(nb_samples=NB_SAMPLES)
    stats.print_timer()
    print("depth : ", stats.depths_info())
    print("branching factor : ", stats.branching_factors_info())
