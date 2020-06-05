from lm_heuristic.tree_search import TreeSearch
from lm_heuristic.tree import Node


def generate_from_grammar(grammar_root: Node, searcher: TreeSearch, nb_tree_walks: int = 500, keep_top_n: int = 1):
    searcher(grammar_root, nb_tree_walks)
    return [str(leaf) for (leaf, _) in searcher.top_n_leaves(keep_top_n)]
