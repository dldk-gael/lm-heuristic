import nltk
from tree_search.tree import Derivation, TreeStats


GRAMMAR_FOLDER = "data/cfg/"
GRAMMAR_NAME = "ex_3"

if __name__ == "__main__":
    # Load grammar
    with open(GRAMMAR_FOLDER + GRAMMAR_NAME + ".cfg") as f:
        str_grammar = f.read()
    grammar = nltk.CFG.fromstring(str_grammar)

    # Prepare grammar tree
    grammar_root = Derivation(grammar.start(), grammar)

    # Compute stats
    stats = TreeStats(grammar_root)
    stats.accumulate_stats(nb_samples=1000)
    print(stats.depths_info())
