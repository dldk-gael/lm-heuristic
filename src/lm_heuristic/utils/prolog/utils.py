from pyswip import Functor


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
