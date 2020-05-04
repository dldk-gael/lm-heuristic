from tree_search.tree import Derivation
from tree_search.evaluation import TreeStats


GRAMMAR_FOLDER = "data/cfg/"
GRAMMAR_NAME = "ex_4"

if __name__ == "__main__":

    # Prepare grammar tree
    grammar_root = Derivation.from_cfg_file(GRAMMAR_FOLDER + GRAMMAR_NAME + '.cfg')

    # Compute stats
    stats = TreeStats(grammar_root)
    stats.accumulate_stats(nb_samples=1000)
    print("depth : ", stats.depths_info())
    print("branching factor : ", stats.branching_factors_info())
