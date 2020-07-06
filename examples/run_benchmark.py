"""
This script shows how to use the evaluation framework
"""

from lm_heuristic.tree_search.mcts import MonteCarloTreeSearch
from lm_heuristic.tree_search.random import RandomSearch
from lm_heuristic.tree_search.evaluator import Evaluator
from lm_heuristic.tree.interface.nltk_grammar import CFGrammarNode
from lm_heuristic.benchmark import Benchmark
from lm_heuristic.sentence_score import GPT2Score


GRAMMAR_FOLDER = "data/cfg/"
GRAMMAR_NAMES = ["ex_1_small", "ex_2_small"]
RESULTS_FOLDER = "results/"
BATCH_SIZE = 1

if __name__ == "__main__":
    # Prepare a toy dataset
    dataset = [
        (CFGrammarNode.from_cfg_file(GRAMMAR_FOLDER + grammar_name + ".cfg"), grammar_name)
        for grammar_name in GRAMMAR_NAMES
    ]

    # Load heuristic function <- GPT2 score
    gpt_2_scorer = GPT2Score("gpt2", length_normalization=True, batch_size=BATCH_SIZE)
    evaluator = Evaluator(gpt_2_scorer)

    # Initialize several strategies
    random_strategy = RandomSearch(evaluator=evaluator, buffer_size=BATCH_SIZE)
    mcts = MonteCarloTreeSearch(evaluator=evaluator, buffer_size=BATCH_SIZE)

    # Run the evaluation and save the results in an csv file
    benchmark = Benchmark(verbose=True)
    results = benchmark([random_strategy, mcts], dataset, nb_tree_walks=[5, 10], nb_random_restarts=2)  # type: ignore
    results.to_csv(RESULTS_FOLDER + "benchmark_results.csv")
