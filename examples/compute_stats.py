from lm_heuristic.tree import CFGrammarNode, TreeStats

"""
Use to evaluate depth and branching factor of tree by performing random sampling on it 
"""

GRAMMAR_FOLDER = "../data/cfg/"
GRAMMAR_NAME = "ex_1_small"

if __name__ == "__main__":

    # Prepare grammar tree
    grammar_root = CFGrammarNode.from_cfg_file(GRAMMAR_FOLDER + GRAMMAR_NAME + '.cfg')

    # Compute stats
    stats = TreeStats(grammar_root)
    stats.accumulate_stats(nb_samples=10000)
    print("depth : ", stats.depths_info())
    print("branching factor : ", stats.branching_factors_info())
