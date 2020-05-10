import random

from lm_heuristic.tree import CFGrammarNode
from lm_heuristic.tree_search import RandomRestartsMCTS, AllocationStrategy
from lm_heuristic.heuristic import Heuristic
from lm_heuristic.sentence_score import GPT2Score


random.seed(3)
GRAMMAR_FOLDER = "data/cfg/"
GRAMMAR_NAME = "ex_1_small"
BATCH_SIZE = 1

if __name__ == "__main__":
    # Load grammar tree
    grammar_root = CFGrammarNode.from_cfg_file(GRAMMAR_FOLDER + GRAMMAR_NAME + ".cfg", shrink=True)

    # Load heuristic function <- GPT2 score
    gpt_2_scorer = GPT2Score("gpt2", batch_size=BATCH_SIZE, length_normalization=True)
    evaluation_fn = lambda terminal_nodes: gpt_2_scorer(list(map(str, terminal_nodes)))
    heuristic = Heuristic(evaluation_fn, use_memory=False)

    # Initialize the search parameters
    meta_mcts = RandomRestartsMCTS(
        heuristic=heuristic,
        buffer_size=BATCH_SIZE,
        nb_random_restarts=3,
        c=1,
        d=1000,
        t=0,
        allocation_strategy=AllocationStrategy.UNIFORM,
        verbose=True,
    )

    # Perform the search and print some info
    best_node = meta_mcts(grammar_root, nb_of_tree_walks=50)
    meta_mcts.print_search_info()
