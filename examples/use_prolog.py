from lm_heuristic.utils.prolog import PrologGrammarEngine
from lm_heuristic.tree.interface.prolog import PrologGrammarNode
from lm_heuristic.tree_search.mcts import MonteCarloTreeSearch
from lm_heuristic.tree_search.evaluator import Evaluator
from lm_heuristic.sentence_score import GPT2Score


GRAMMAR_FOLDER = "data/fcfg/"
GRAMMAR_NAME = "feat0"
BATCH_SIZE = 1

if __name__ == "__main__":

    # Initialize the prolog engine
    prolog_engine = PrologGrammarEngine("prolog/methods.pl")
    grammar_root = PrologGrammarNode.from_cfg_file(prolog_engine, path=GRAMMAR_FOLDER + GRAMMAR_NAME + ".fcfg")

    prolog_engine.set_random_seed(1)
    # Load heuristic function <- GPT2 score
    gpt_2_scorer = GPT2Score("gpt2", batch_size=BATCH_SIZE, length_normalization=True)
    evaluator = Evaluator(lambda terminal_nodes: gpt_2_scorer(list(map(str, terminal_nodes))))

    # Initialize the search parameters
    mcts = MonteCarloTreeSearch(evaluator=evaluator, progress_bar=True,)

    # Perform the search and print some info
    best_node, best_value = mcts.search(grammar_root, nb_of_tree_walks=15)
