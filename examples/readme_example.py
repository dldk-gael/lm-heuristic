from lm_heuristic.tree.interface import nltk_grammar
from lm_heuristic.sentence_score import GPT2Score
from lm_heuristic.tree_search import Evaluator
from lm_heuristic.tree_search.mcts import (
    MonteCarloTreeSearch,
    AllocationStrategy,
    RessourceDistributor,
    standart_ucb,
)

GRAMMAR = """
s -> np vp 
np -> 'Gael'
vp -> v obj 
v -> 'knows' | 'know'
obj -> 'Bas' | 'him'
"""

# Generate a tree from a context free grammar
grammar_root = nltk_grammar.CFGrammarNode.from_string(GRAMMAR)
print(grammar_root.children()[0])  # np vp.

# Initialize Sentence scorer
gpt2_scorer = GPT2Score(model_name="gpt2", batch_size=2, length_normalization=True)
gpt2_scorer.build()  # load GPT2 in memory
print(gpt2_scorer(["Gael knows Bas", "Gael know Bas"]))  # => 6.1e-06, 3.6e-06

# Initialize MCTS to search the best leaf of the grammar tree
mcts = MonteCarloTreeSearch(
    evaluator=Evaluator(gpt2_scorer),  # add memoization on top of the sentence scorer
    buffer_size=1,  # to input sentence by batch to the scorer
    ressource_distributor=RessourceDistributor(AllocationStrategy.ALL_FROM_ROOT),
    nb_random_restarts=1,
    ucb_function=standart_ucb,  # specify the selection policy
    parallel_strategy="none",  # use multiprocess to run the evaluation in another process
    progress_bar=True,  # to plot a progress bar
)

# context will be concatenated to the left side of each input sentences
gpt2_scorer.set_context("Who knows Bas ?")  

best_leaf, best_value = mcts.search(grammar_root, nb_of_tree_walks=10)
# => 'Gael knows him.", "4.06e-4"
