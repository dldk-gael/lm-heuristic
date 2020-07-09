import pickle

productions = pickle.load(open("data/brown_rules.pickle", "rb"))

print(len(productions))