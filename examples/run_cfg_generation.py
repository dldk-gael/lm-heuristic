from lm_heuristic.tree_search import RandomSearch
from lm_heuristic.sentence_score import GPT2Score
from lm_heuristic.heuristic import Heuristic
from lm_heuristic.generation import GenerateFromCFG

GRAMMAR_FOLDER = "data/cfg/"
GRAMMAR_NAME = "ex_4"

if __name__ == "__main__":
    gpt_2_scorer = GPT2Score("gpt2", length_normalization=True, batch_size=1)
    evaluation_fct = lambda terminal_nodes: gpt_2_scorer(list(map(str, terminal_nodes)))
    heuristic = Heuristic(evaluation_fct)
    searcher = RandomSearch(heuristic)

    generator = GenerateFromCFG(searcher, nb_tree_walks=50)
    outputs = generator(GRAMMAR_FOLDER + GRAMMAR_NAME + ".cfg", nb_samples=5)

    for i, (sentence, score) in enumerate(outputs):
        print("n°%d (score = %f) : %s" % (i, score, sentence))
