from lm_heuristic.tree import FeatureGrammarNode
from nltk.featstruct import FeatStruct, rename_variables
from nltk.sem import Variable
from lm_heuristic.tree_search import RandomSearch
from lm_heuristic.heuristic import Heuristic


GRAMMAR_FOLDER = "data/fcfg/"
GRAMMAR_NAME = "toy"


def evaluation_fn_one_node(node):
    if str(node) == "DEAD_END":
        return 0
    return 1


def evaluation_fn(nodes):
    return [evaluation_fn_one_node(node) for node in nodes]


if __name__ == "__main__":
    grammar_root = FeatureGrammarNode.from_fcfg_file(GRAMMAR_FOLDER + GRAMMAR_NAME + ".fcfg")
    for i in range(5):
        print(grammar_root.find_random_valid_leaf())

    if False:
        print(grammar_root.find_random_valid_leaf())

        heuristic = Heuristic(evaluation_fn)

        # Initialize the search parameters
        searcher = RandomSearch(
            heuristic=heuristic,
            buffer_size=1,

        )

        # Perform the search and print some info
        searcher(grammar_root, nb_of_tree_walks=50)
        searcher.print_search_info()
