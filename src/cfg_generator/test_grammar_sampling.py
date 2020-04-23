import nltk
from nltk.parse.generate import generate, demo_grammar

GRAMMAR_FOLDER = '../../data/utils.py/'
if __name__ == '__main__':
    with open(GRAMMAR_FOLDER+'bas.utils.py') as f:
        str_grammar = f.read()
    grammar = nltk.FeatureGrammar.fromstring(str_grammar)

    for sentence in generate(grammar):
        print(" ".join(sentence))
