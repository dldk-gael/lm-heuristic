import nltk
from nltk.parse.generate import generate


def generate_all_sentences(grammar):
    """
    grammar : file containing the grammar rules (.cfg)
    return : list[str] all the sentence that can been derivated from the grammar
    """
    with open(grammar) as f:
        str_grammar = f.read()
    grammar = nltk.CFG.fromstring(str_grammar)
    return list(map(lambda l: " ".join(l), generate(grammar)))
