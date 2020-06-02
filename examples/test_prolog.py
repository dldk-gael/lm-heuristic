from nltk.grammar import FeatureGrammar
from lm_heuristic.prolog.parser import ParseToProlog

GRAMMAR_FOLDER = "data/fcfg/"
GRAMMAR_NAME = "toy"

with open(GRAMMAR_FOLDER + GRAMMAR_NAME + ".fcfg") as file:
    str_grammar = file.read()
    
prolog_predicates = ParseToProlog(feature_grammar=True)(str_grammar)
for predicate in prolog_predicates:
    print("%s."%predicate)
