from nltk.corpus import brown
from nltk.grammar import Production, Nonterminal
import pickle 

tag_words = dict()

for word, pos in brown.tagged_words():
    tag_words.setdefault(word, set())
    raw_pos = pos.split("-")[0] # remove -TL or -NL information
    tag_words[word].add(raw_pos)

production_dict = {}
for word in tag_words.keys():
    production_dict[word] = []
    for pos in tag_words[word]:
        new_prod = Production(lhs=Nonterminal(pos), rhs=(word,))
        production_dict[word].append(new_prod)

pickle.dump(production_dict, open("brown_pos_tag.pickle", "wb"))