from lm_heuristic.prolog.utils import convert_grammar_to_prolog
from nltk.grammar import CFG


GRAMMAR_FOLDER = "data/cfg/"
GRAMMAR_NAME = "ex_1_small"

with open(GRAMMAR_FOLDER + GRAMMAR_NAME + ".cfg") as file:
    str_grammar = file.read()
    str_nltk_grammar = str(CFG.fromstring(str_grammar))
    predicates = convert_grammar_to_prolog(str_nltk_grammar)

for predicate in predicates:
    print("%s."%predicate)