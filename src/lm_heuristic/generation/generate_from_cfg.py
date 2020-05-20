from lm_heuristic.tree_search import TreeSearch
from lm_heuristic.tree import CFGrammarNode


def generate_from_cfg(grammar_root: CFGrammarNode, searcher: TreeSearch, nb_tree_walks: int = 500, keep_top_n: int = 1):
    searcher(grammar_root, nb_tree_walks)
    return [str(leaf) for (leaf, _) in searcher.top_n_leaves(keep_top_n)]


class GenerateFromCFG:
    def __init__(self, searcher: TreeSearch, nb_tree_walks: int = 500):
        self.searcher = searcher
        self.nb_tree_walks = nb_tree_walks

    def __call__(
        self, path_to_grammar: str, nb_samples: int = 1,
    ):
        grammar_root = CFGrammarNode.from_cfg_file(path_to_grammar)
        self.searcher(grammar_root, self.nb_tree_walks)

        return [(str(leaf), value) for (leaf, value) in self.searcher.top_n_leaves(nb_samples)]
