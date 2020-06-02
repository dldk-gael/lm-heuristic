from typing import List, Dict, Union, Set
from nltk.grammar import FeatStructNonterminal, Production, FeatureGrammar, Nonterminal
from nltk.featstruct import Feature
from nltk.sem import Variable


def parse_to_prolog(grammar_str: str) -> List[str]:
    """
    Use to transform NLTK grammar into prolog predicates.
    Support both feature grammar and vanilla context free grammar.

    Grammar that has the form :
        # Rules
        s -> np vp
        np -> n | det n
        ...
        # Lexicon
        v -> 'want' | 'wants'

    Will be transform to a list of prolog predicates
        rule(s, [np, vp])
        rule(np, [n])
        rule(np, [det, n])
        ...
        rule(v, ['want'])
        terminal('want)
    """
    nltk_featgrammar = FeatureGrammar.fromstring(grammar_str)
    features_dict = compute_features_dict(nltk_featgrammar.productions())

    pl_predicates = []

    for production in nltk_featgrammar.productions():
        lhs = parse_term(production.lhs(), features_dict)
        rhs = [parse_term(term, features_dict) for term in production.rhs()]
        pl_predicates.append("rule(%s, [%s])" % (lhs, ", ".join(rhs)))

    terminal_symbols = compute_terminal_symbols(nltk_featgrammar.productions())
    for terminal in terminal_symbols:
        pl_predicates.append("terminal(%s)" % terminal.lower())

    return pl_predicates


def compute_features_dict(productions: List[Production]) -> Dict[str, List[str]]:
    features_dict: Dict[str, List[str]] = dict()
    for production in productions:
        for term in (production.lhs(),) + production.rhs():
            if isinstance(term, Nonterminal):
                name = str(term[Feature("type")]).lower()
                features_dict.setdefault(name, [])
                current_features = set(features_dict[name])
                new_features = set(str(feature) for feature in term if str(feature) != "*type*")
                features_dict[name] = list(current_features.union(new_features))

    return features_dict


def parse_term(term: Union[FeatStructNonterminal, str], features_dict: Dict[str, List[str]]) -> str:
    if not isinstance(term, Nonterminal):
        return term.lower()

    name = str(term[Feature("type")]).lower()
    param_list = []
    for feature in features_dict[name]:
        if feature in term:
            param_list.append(transform_value(term[feature]))
        else:
            param_list.append("_")
    return "%s(%s)" % (name, ", ".join(param_list)) if len(param_list) != 0 else name


def transform_value(value):
    if isinstance(value, Variable):
        # Then, the variable is on the form ?x_1
        # We transform that to X_1
        return str(value)[1:].upper()
    else:
        return str(value).lower()


def compute_terminal_symbols(productions: List[Production]) -> Set[str]:
    terminal_symbols = set()
    for production in productions:
        for term in production.rhs():
            if not isinstance(term, Nonterminal):
                terminal_symbols.add(term.lower())
    return terminal_symbols
