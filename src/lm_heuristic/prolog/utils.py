import re 
from pyswip import Functor

def convert_grammar_to_prolog(grammar_str):
    pl_predicates = []

    for lign in grammar_str.split("\n")[1:]:
        rule = lign.split("->")
        lhs = rule[0].strip()
        rhs = rule[1].strip().split(" ")
        rhs = [x.strip() for x in rhs]
        pl_predicates.append("rule(%s, [%s])" % (lhs, ", ".join(rhs)))

    for terminal in set(re.findall(r"'\w+'", grammar_str)):
        pl_predicates.append("terminal(%s)" % terminal)

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
