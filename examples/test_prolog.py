from lm_heuristic.prolog.parser import parse_to_prolog

GRAMMAR_FOLDER = "data/fcfg/"
GRAMMAR_NAME = "toy"

with open(GRAMMAR_FOLDER + GRAMMAR_NAME + ".fcfg") as file:
    str_grammar = file.read()

prolog_predicates = parse_to_prolog(str_grammar)
for predicate in prolog_predicates:
    print("%s." % predicate)
