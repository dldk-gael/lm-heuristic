from nltk.parse import CoreNLPParser
from nltk.corpus import brown
import pickle
from tqdm import tqdm


def number_of_words(sentence):
    """
    Do not count punctuation as word
    """
    return len([word for word in sentence if word.isalnum()])


def is_rhs_terminal(production):
    rhs = production.rhs()
    return len(rhs) == 1 and isinstance(rhs[0], str)


parser = CoreNLPParser(url="http://localhost:9000")

sentences = brown.sents()

# FILTER SHORT AND LONG SENTENCES
filter_sentences = []
for sentence in tqdm(sentences):
    nb_words = number_of_words(sentence)
    if nb_words >= 5 and nb_words <= 10:
        filter_sentences.append(sentence)

# PARSE SENTENCES
productions = []
for sentence in tqdm(filter_sentences):
    parse_tree = next(iter(parser.parse(sentence)))
    productions += parse_tree.productions()

unique_productions = list(set(productions))

# REMOVE TERMINAL SYMBOLS
productions_wo_term = []
for prod in unique_productions:
    if not is_rhs_terminal(prod):
        productions_wo_term.append(prod)

pickle.dump(productions_wo_term, open("brown_rules.pickle", "wb"))
