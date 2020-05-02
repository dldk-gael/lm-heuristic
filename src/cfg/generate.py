import nltk
from nltk.parse.generate import generate
import argparse


def generate_all_sentences(grammar):
    """
    grammar : file containing the grammar rules (.cfg)
    return : list[str] all the sentence that can been derivated from the grammar
    """
    with open(grammar) as f:
        str_grammar = f.read()
    grammar = nltk.CFG.fromstring(str_grammar)
    return list(map(lambda l: " ".join(l), generate(grammar)))


"""
Script to quickly test if a grammar is correct at construction time 
ie: can sentences be derivated from the grammar? if yes, how many ? 
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_file")
    args = parser.parse_args()
    all_sentences = generate_all_sentences(args.path_file)
    print("Number of sentences : ", len(all_sentences))
