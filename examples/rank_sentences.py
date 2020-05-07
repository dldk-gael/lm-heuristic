import nltk
from nltk.parse.generate import generate
from lm_heuristic.sentence_score import GPT2Score
import os
import pickle

"""
This script :
1/ generates all possible sentences from a CFG, 
2/ rank all sentences using a gpt2 scorer,
3/ print the X best results

Should only be used with CFG that can generate only small amount of sentences 
"""

GRAMMAR_FOLDER = "../data/cfg/"
RESULT_FOLDER = "../results/"
GRAMMAR_NAME = "bas"
X_BEST_RESULTS = 100


def generate_all_sentences(grammar):
    """
    grammar : file containing the grammar rules (.cfg)
    return : list[str] all the sentence that can been derivated from the grammar
    """
    with open(grammar) as f:
        str_grammar = f.read()
    grammar = nltk.CFG.fromstring(str_grammar)
    return list(map(lambda l: " ".join(l), generate(grammar)))


if __name__ == "__main__":

    if not os.path.exists(RESULT_FOLDER + GRAMMAR_NAME + ".pkl"):
        sentences = generate_all_sentences(GRAMMAR_FOLDER + GRAMMAR_NAME + ".cfg")
        gpt2_scorer = GPT2Score("gpt2", length_normalization=True, verbose=True)
        ranked_sentences = gpt2_scorer.rank_sentences(sentences)
        pickle.dump(ranked_sentences, open(RESULT_FOLDER + GRAMMAR_NAME + ".pkl", "wb"))

    ranked_sentences = pickle.load(open(RESULT_FOLDER + GRAMMAR_NAME + ".pkl", "rb"))
    GPT2Score.print_ranked_sentences(ranked_sentences, n=X_BEST_RESULTS)
