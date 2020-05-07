from lm_heuristic.tree_search.random import RandomSearch
from lm_heuristic.tree_search.mcts import MonteCarloTreeSearch
from lm_heuristic.tree import CFGrammarNode
from lm_heuristic.evaluation import EvalStrategy
from lm_heuristic.heuristic import Heuristic
from lm_heuristic.heuristic.sentence_score import GPT2Score

"""
This script shows how to use the evaluation framework 
"""

GRAMMAR_FOLDER = "../data/cfg/"
GRAMMAR_NAMES = ["ex_1_small", "ex_2_small"]

if __name__ == "__main__":
    # Prepare a toy dataset
    dataset = [CFGrammarNode.from_cfg_file(GRAMMAR_FOLDER + grammar_name + ".cfg") for grammar_name in GRAMMAR_NAMES]

    # Load heuristic function <- GPT2 score
    gpt_2_scorer = GPT2Score("gpt2", length_normalization=True, batch_size=16)
    eval_fn = lambda terminal_nodes: gpt_2_scorer(list(map(str, terminal_nodes)))
    heuristic = Heuristic(eval_fn)

    # Initialize several strategies
    random_strategy = RandomSearch(heuristic=heuristic)
    mcts = MonteCarloTreeSearch(heuristic=heuristic)

    # Initialize the evaluation framework
    evaluate = EvalStrategy(verbose=True)
    evaluate([random_strategy, mcts], dataset, nb_tree_walks=[5, 10], nb_random_restarts=2)
    print(evaluate.eval_results)
