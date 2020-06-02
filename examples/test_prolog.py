from lm_heuristic.prolog.parser import parse_to_prolog

GRAMMAR_FOLDER = "data/cfg/"
GRAMMAR_NAME = "ex_1_large"

with open(GRAMMAR_FOLDER + GRAMMAR_NAME + ".cfg") as file:
    str_grammar = file.read()

prolog_predicates = parse_to_prolog(str_grammar)
for predicate in prolog_predicates:
    print("%s." % predicate)
