from nltk.parse import CoreNLPParser

parser = CoreNLPParser(url='http://localhost:9000')

print(list(parser.tokenize("I like tennis.")))

results = list(parser.parse('I like tennis.'.split()))[0]
productions = results.productions()
print(productions)