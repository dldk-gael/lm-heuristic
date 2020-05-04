from tree_search.strategy import RandomSearch, MonteCarloTreeSearch
from tree_search.tree import Derivation
from tree_search.evaluation import EvalStrategy
from heuristic import GPT2Score

"""
This script shows how to use a random searcher 
"""

GRAMMAR_FOLDER = 'data/cfg/'
GRAMMAR_NAME = 'ex_2'

if __name__ == '__main__':
    # Prepare root node
    root = Derivation.from_cfg_file(GRAMMAR_FOLDER + GRAMMAR_NAME + '.cfg')

    # Load heuristic function <- GPT2 score
    gpt_2_scorer = GPT2Score('gpt2', length_normalization=True, batch_size=16)
    heuristic = lambda terminal_nodes: gpt_2_scorer(list(map(str, terminal_nodes)))

    # Initialize a random strategy
    random_strategy = RandomSearch(evaluation_fn=heuristic, n_samples=100)

    # Initialize the evaluation framework
    evaluate = EvalStrategy(verbose=True)
    evaluate(random_strategy, root, nb_random_restarts=2, nb_of_tree_walks=10)
    print(evaluate.eval_results)
