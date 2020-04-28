from tree_search.tree import Derivation
from tree_search.strategy import MonteCarloTreeSearch
import nltk
import logging
from heuristic import GPT2Score

logging.basicConfig(level=logging.DEBUG)


GRAMMAR_FOLDER = 'data/cfg/'
if __name__ == '__main__':
    # Load grammar
    with open(GRAMMAR_FOLDER+'ex_2.cfg') as f:
        str_grammar = f.read()
    grammar = nltk.CFG.fromstring(str_grammar)

    # Load heuristic function <- GPT2 score
    gpt_2_scorer = GPT2Score('gpt2')
    heuristic = lambda terminal_nodes: gpt_2_scorer(list(map(str, terminal_nodes)))

    # heuristic = lambda terminal_nodes: [0] * len(terminal_nodes)

    # Prepare root node
    root = Derivation(grammar.start(), grammar)

    # Initialize and perform the search
    mtcs = MonteCarloTreeSearch(root, evaluation_fn=heuristic, batch_size=16,
                                nb_of_tree_walks=100, c=1, d=1000, t=0)

    best_node = mtcs.search()
