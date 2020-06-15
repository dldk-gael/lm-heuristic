import random
from nltk.grammar import FeatureGrammar, FeatStructNonterminal, Production
from nltk.featstruct import FeatStruct
from nltk.sem import Variable

from lm_heuristic.tree import FeatureGrammarNode

def format_list(obj_list):
    obj_list = obj_list if isinstance(obj_list, list) else [obj_list]
        
    if len(obj_list) == 0:
        return "na"
    else:
        tail = format_list(obj_list[1:])
        return FeatStructNonterminal(head=obj_list[0], tail=tail)


# INPUT DATA
Gael = FeatStruct(proper="Gael", is_proper=True, is_noun=False, gender="male", form="singular")
Bas = FeatStruct(proper="Bas", is_proper=True, is_noun=False, gender="male", form="singular")
club = FeatStruct(proper="na", is_proper=False, is_noun=True, noun="club", gender="neuter", form="singular")
know = FeatStruct(sem="know", sym=True)
member = FeatStruct(sem="member", sym=False)
singular = FeatStruct(form='singular')
plural = FeatStruct(form='plural')

# Dynamic rules to output proper name
dynamic_productions = [
    Production(FeatStructNonterminal("ProperName[proper=Bas]"), ["Bas"]),
    Production(FeatStructNonterminal("ProperName[proper=Gael]"), ["Gael"]),
    Production(FeatStructNonterminal("Noun[noun=club, form=singular]"), ["club"]),
]

# know(Gael, Bas)
know_gael_bas = FeatStruct(arg0=format_list(Gael), pred=know.unify(singular), arg1=format_list(Bas))

# member([Gael, Bas], club)
member_gael_club = FeatStruct(arg0=format_list(Gael), pred=member.unify(singular), arg1=format_list(club))

list_of_facts = format_list([know_gael_bas, member_gael_club]).unify(FeatStructNonterminal("Facts"))
list_of_facts.freeze()

root = FeatStructNonterminal("Root")
start_production = Production(lhs=root, rhs=[list_of_facts])
with open("data/fcfg/foaf_grammar.fcfg", "r") as grammar_file:
    grammar_str = grammar_file.read()
    static_grammar = FeatureGrammar.fromstring(grammar_str)

all_productions = static_grammar.productions() + [start_production] + dynamic_productions
dynamic_grammar = FeatureGrammar(start=root, productions=all_productions)

print(dynamic_grammar)
feature_root = FeatureGrammarNode(symbols=root, feature_grammar=dynamic_grammar)
random.seed(1)
for _ in range(15):
    print(feature_root.find_random_valid_leaf_previous())
