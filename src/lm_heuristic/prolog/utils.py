from typing import List
import re
from pyswip import Functor


def parse_to_prolog(grammar_str, feature_grammar: bool = False) -> List[str]:
    if feature_grammar:
        return []
    else:
        return parse_CFG_str_grammar(grammar_str)


def parse_CFG_str_grammar(grammar_str: str) -> List[str]:
    """
    Use to parse grammar that as the form
        # Rules
        s -> np vp
        np -> n | det n
        ...
        # Lexicon
        v -> 'want' | 'wants'

    Into a list of prolog predicates
        rule(s, [np, vp])
        rule(np, [n])
        rule(np, [det, n])
        ...
        rule(v, ['want'])
        terminal('want)
    """

    pl_predicates = []

    # First lign on NLTK grammar is comment lign
    for line in grammar_str.split("\n"):
        # We skip line that do not contain an '->'
        if "->" in line:
            # All need to be lower case so that prolog do not consider term as Variable
            line = line.lower()
            rule = line.split("->")
            lhs = rule[0].strip()
            rhs = rule[1].strip().split(" ")
            rhs = [x.strip() for x in rhs]
            pl_predicates.append("rule(%s, [%s])" % (lhs, ", ".join(rhs)))

    for terminal in set(re.findall(r"'\w+'", grammar_str)):
        pl_predicates.append("terminal(%s)" % terminal.lower())

    return pl_predicates



def format_value(value):
    output = ""
    if isinstance(value, list):
        output = "[" + ", ".join([format_value(val) for val in value]) + "]"
    elif isinstance(value, Functor) and value.arity == 2:
        output = "{0}{1}{2}".format(value.args[0], value.name, value.args[1])
    else:
        output = "{}".format(value)

    return output


def format_term(term):
    if isinstance(term, list):
        return [format_term(x) for x in term]
    else:
        return term.value


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def join(symbols):
    """
    Use to join symbol with "," and add "'" arround number
    """
    if len(symbols) == 0:
        return ""
    virgule = ", " if len(symbols) > 1 else ""
    symbol = "'%s'" % symbols[0] if is_number(symbols[0]) else symbols[0]
    return symbol + virgule + join(symbols[1:])
