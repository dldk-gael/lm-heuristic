import nltk
from nltk.parse.generate import generate

GRAMMAR_FOLDER = 'data/cfg/'

if __name__ == '__main__':
    with open(GRAMMAR_FOLDER+'ex.fcfg') as f:
        str_grammar = f.read()
    grammar = nltk.grammar.FeatureGrammar.fromstring(str_grammar)
    sentences = list(generate(grammar))

    for sentence in sentences:
        print(sentence)

    start = grammar.start()
    next = grammar.productions(lhs=start)
    production_0 = next[0]
    t = production_0.lhs()
    fin = 0