from tree_search.tree import Derivation
from tree_search.strategy import MonteCarloTreeSearch
import nltk
from heuristic import GPT2Score
import random


random.seed(3)
GRAMMAR_FOLDER = 'data/cfg/'
GRAMMAR_NAME = 'ex_2'

BATCH_SIZE = 16  #

if __name__ == '__main__':
    # Load grammar
    with open(GRAMMAR_FOLDER + GRAMMAR_NAME + '.cfg') as f:
        str_grammar = f.read()
    grammar = nltk.CFG.fromstring(str_grammar)

    # Load heuristic function <- GPT2 score
    gpt_2_scorer = GPT2Score('gpt2', batch_size=BATCH_SIZE, length_normalization=True)
    heuristic = lambda terminal_nodes: gpt_2_scorer(list(map(str, terminal_nodes)))

    # Prepare grammar tree
    root = Derivation(grammar.start(), grammar)

    # Initialize the search parameters
    mtcs = MonteCarloTreeSearch(root, evaluation_fn=heuristic, batch_size=BATCH_SIZE,
                                nb_of_tree_walks=20, c=1, d=1000, t=0)

    # Perform the search and print some info
    best_node = mtcs.search()
    mtcs.print_search_info()

    # Example on how to print more info on internal node located on the best path
    mtcs.counter_path()[1].detailed_node_info()

