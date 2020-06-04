from lm_heuristic.tree import FeatureGrammarNode
from nltk.featstruct import FeatStruct, rename_variables
from nltk.sem import Variable
from lm_heuristic.tree_search import MonteCarloTreeSearch, AllocationStrategy
from lm_heuristic.heuristic import Heuristic


GRAMMAR_FOLDER = "data/fcfg/"
GRAMMAR_NAME = "feat0"


def evaluation_fn_one_node(node):
    if str(node) == "DEAD_END":
        return 0
    return 1


def evaluation_fn(nodes):
    return [evaluation_fn_one_node(node) for node in nodes]


if __name__ == "__main__":
    grammar_root = FeatureGrammarNode.from_cfg_file(GRAMMAR_FOLDER + GRAMMAR_NAME + ".fcfg")
  
    heuristic = Heuristic(evaluation_fn)

    # Initialize the search parameters
    mcts = MonteCarloTreeSearch(
        heuristic=heuristic,
        buffer_size=1,
        c=1,
        d=1000,
        t=0,
        stats_samples=100,
        allocation_strategy=AllocationStrategy.UNIFORM,
        verbose=True,
    )

    # Perform the search and print some info
    best_node = mcts(grammar_root, nb_of_tree_walks=15)
    mcts.print_search_info()
