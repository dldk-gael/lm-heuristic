import nltk
from nltk.parse.generate import generate

if __name__ == '__main__':
    with open('data/cfg/ex_1.cfg') as f:
        str_grammar = f.read()
    grammar = nltk.CFG.fromstring(str_grammar)
    print(len(list(generate(grammar))))
    #for sentence in generate(grammar):
    #    print(" ".join(sentence))

    print(grammar)