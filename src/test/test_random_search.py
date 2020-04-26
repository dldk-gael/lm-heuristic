from tree_search.strategy import RandomSearch
from tree_search.tree import Derivation
import nltk

from heuristic import GPT2Score

GRAMMAR_FOLDER = 'data/cfg/'
if __name__ == '__main__':
    # Load grammar
    with open(GRAMMAR_FOLDER+'ex_1.cfg') as f:
        str_grammar = f.read()
    grammar = nltk.CFG.fromstring(str_grammar)

    # Load heuristic function <- GPT2 score
    gpt_2_scorer = GPT2Score('gpt2')
    heuristic = lambda terminal_nodes: gpt_2_scorer(list(map(str, terminal_nodes)))

    # Prepare root node
    root = Derivation(grammar.start(), grammar)

    # Initialize and perform the search
    random_search = RandomSearch(root, evaluation_fn=heuristic, n_samples=100, batch_size=16)
    final_derivation = random_search.search()

    print(final_derivation)
    print("\nPath")
    print("\n".join(map(str, random_search.path())))