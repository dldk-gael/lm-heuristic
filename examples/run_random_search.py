from src.tree_search import RandomSearch
from src.tree import CFGrammarNode
from src.heuristic import Heuristic
from src.heuristic.sentence_score import GPT2Score

"""
This script shows how to use a random searcher 
"""

GRAMMAR_FOLDER = "../data/cfg/"
GRAMMAR_NAME = "ex_4"

if __name__ == "__main__":
    # Prepare root node
    root = CFGrammarNode.from_cfg_file(GRAMMAR_FOLDER + GRAMMAR_NAME + ".cfg")

    # Load heuristic function <- GPT2 score
    gpt_2_scorer = GPT2Score("gpt2", length_normalization=True, batch_size=1)
    heuristic = Heuristic(
        evaluation_fct=lambda terminal_nodes: gpt_2_scorer(
            list(map(str, terminal_nodes))
        )
    )

    # Initialize and perform the search
    random_search = RandomSearch(heuristic=heuristic)
    final_derivation = random_search(root, nb_of_tree_walks=10)

    random_search.print_search_info()

