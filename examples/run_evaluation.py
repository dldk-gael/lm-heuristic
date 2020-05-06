from lm_heuristic.tree_search.random import RandomSearch
from lm_heuristic.tree_search.mcts import MonteCarloTreeSearch, AllocationStrategy
from lm_heuristic.tree import CFGrammarNode
from lm_heuristic.evaluation import EvalStrategy
from lm_heuristic.heuristic import Heuristic
from lm_heuristic.heuristic.sentence_score import GPT2Score

"""
This script shows how to use the evaluation framework 
"""

GRAMMAR_FOLDER = "../data/cfg/"
GRAMMAR_NAME = "ex_2"

if __name__ == "__main__":
    # Prepare root node
    root = CFGrammarNode.from_cfg_file(GRAMMAR_FOLDER + GRAMMAR_NAME + ".cfg")

    # Load heuristic function <- GPT2 score
    gpt_2_scorer = GPT2Score("gpt2", length_normalization=True, batch_size=16)
    eval_fn = lambda terminal_nodes: gpt_2_scorer(list(map(str, terminal_nodes)))
    heuristic = Heuristic(eval_fn)

    # Initialize a random strategy
    random_strategy = RandomSearch(heuristic=heuristic, buffer_size=1, verbose=False)

    # Initialize the evaluation framework
    evaluate = EvalStrategy(verbose=True)
    evaluate(random_strategy, root, nb_tree_walks=[5, 10], nb_random_restarts=2)
    print(evaluate.eval_results)
