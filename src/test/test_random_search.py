from tree_search.strategy import RandomSearch
from tree_search.tree import Derivation
import nltk

GRAMMAR_FOLDER = 'data/cfg/'
if __name__ == '__main__':
    with open(GRAMMAR_FOLDER+'ex_1.cfg') as f:
        str_grammar = f.read()
    grammar = nltk.CFG.fromstring(str_grammar)

    root = Derivation((grammar.start(),), grammar)
    random_search = RandomSearch(root)

    final_derivation = random_search.search()
    print(final_derivation)
    print("Path")
    print("\n".join(map(str, random_search.path())))