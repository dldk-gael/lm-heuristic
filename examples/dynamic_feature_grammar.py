import random
from typing import *
from nltk.grammar import FeatureGrammar, FeatStructNonterminal, Production
from nltk.featstruct import FeatStruct

from lm_heuristic.tree.interface.nltk_grammar import FeatureGrammarNode


def format_list(obj_list: List[FeatStruct]) -> FeatStruct:
    """
    :param obj_list. List of feature structure to chain inside a nested feature structure.
    ex: if we have [arg0, arg1, arg2], it will return the following feature structure
        ->  [head=arg0, tail=[head=arg1, tail=[head=arg2, tail=na]]]
    """
    obj_list = obj_list if isinstance(obj_list, list) else [obj_list]

    if len(obj_list) == 0:
        return "na"
    else:
        tail = format_list(obj_list[1:])
        return FeatStructNonterminal(head=obj_list[0], tail=tail)


def format_list_of_facts(facts_list: List[FeatStruct]) -> FeatStructNonterminal:
    """
    1/ Embed the list of facts in a nested feature struct (using in format_list)
    2/ Embed it in a Nonterminal symbol that have the name facts
    3/ Freeze the structure
    """
    # TODO: check why we have to freeze the structure
    feature_lists = format_list(facts_list).unify(FeatStructNonterminal("Facts"))
    feature_lists.freeze()
    return feature_lists

def proper_rule(person: FeatStruct) -> Production:
    """
    :person : feature structure that characterize one person and containt a proper attribute
    :return : the production rule that can generate the propernoun
                ex: "ProperName[proper=Bas] -> "Bas"
    """
    return Production(FeatStructNonterminal("ProperName[proper=%s]" % person["proper"]), [person["proper"]])


def noun_rule(entity_object: FeatStruct) -> Production:
    # TODO: add the feature to put different name (for plural / singular / synonyme)
    return Production(FeatStructNonterminal("Noun[sem=%s]" % entity_object["noun"]), [entity_object["noun"]])


def entity_specific_rule(entity: FeatStruct) -> Production:
    if entity["is_proper"]:
        return proper_rule(entity)
    if entity["is_noun"]:
        return noun_rule(entity)
    raise ValueError("entity can not be represented by a noun, neither a propername")


def create_fact(
    pred: FeatStruct, arg_0: Union[List[FeatStruct], FeatStruct], arg_1: Union[List[FeatStruct], FeatStruct]
) -> FeatStruct:
    """
    Construct feature structure than represent fact. 
    ex: know(Gael, [Bas, Justine]), return a feature structure of the type :
        [pred=[sem=know],
         arg0 = [head=Gael, tail=na],
         arg1 = [head=Bas, tail=[head=Justine, tail=na]]]
    """
    arg_0 = arg_0 if isinstance(arg_0, list) else [arg_0]
    arg_1 = arg_1 if isinstance(arg_1, list) else [arg_1]
    return FeatStruct(arg0=format_list(arg_0), arg1=format_list(arg_1), pred=pred)


# STATIC LIST OF ALL PREDICATES
know = FeatStruct(sem="know")
know_r = FeatStruct(sem="know_r")
member = FeatStruct(sem="member")

# INPUT DATA
Gael = FeatStruct(proper="Gael", is_proper=True, explicit=True, is_noun=False, gender="male", form="singular")
Bas = FeatStruct(proper="Bas", is_proper=True, explicit=True, is_noun=False, gender="male", form="singular")
Justine = FeatStruct(proper="Justine", is_proper=True, explicit=True, is_noun=False, gender="female", form="singular")
club = FeatStruct(
    proper="na", is_proper=False, is_noun=True, explicit=True, noun="tennis_club", gender="neuter", form="singular"
)

# Dynamic rules that are specific to this input data
# for instance "ProperName[proper=Bas] -> "Bas"
dynamic_productions = [entity_specific_rule(entity) for entity in [Gael, Bas, Justine, club]]

# SOME FACTS AS EXAMPLE :
know_gael_bas_justine = create_fact(know, Gael, [Bas, Justine]) # know(Gael, [Bas, Justine])
know_gael_bas = create_fact(know, Gael, Bas) # know(Gael, Bas)
know_gael_justine = create_fact(know, Gael, Justine) # know(Gael, Justine)
know_justine_bas = create_fact(know, Justine, Bas) # know(Justine, Bas)
member_bas_club = create_fact(member, Bas, club) # member(Bas, club)
member_gael_bas_club = create_fact(member, [Gael, Bas], club) # member([Gael, Bas], club)

# LIST OF FACT THAT WILL BE INPUT TO THE GRAMMAR :
list_of_facts = format_list_of_facts([know_gael_justine, member_gael_bas_club])
root = FeatStructNonterminal("Root")
start_production = Production(lhs=root, rhs=[list_of_facts])

# CREATE A NEW GRAMMAR THAT COMBINE THE STATIC + DYNAMIC RULES
with open("data/fcfg/feature_grammar.fcfg", "r") as grammar_file:
    grammar_str = grammar_file.read()
    static_grammar = FeatureGrammar.fromstring(grammar_str)

all_productions = static_grammar.productions() + [start_production] + dynamic_productions
dynamic_grammar = FeatureGrammar(start=root, productions=all_productions)

print(dynamic_grammar)

# SAMPLE SOME LEAF FROM THE ROOT
feature_root = FeatureGrammarNode(symbols=root, feature_grammar=dynamic_grammar)

random.seed(2)
# print(feature_root.find_random_valid_leaf_debug())

for i in range(10):
    random.seed(i)
    print(feature_root.find_random_valid_leaf())
