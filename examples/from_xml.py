from lm_heuristic.tree.interface.xml import XMLGrammarNode
from lm_heuristic.tree.stats import TreeStats

root_node = XMLGrammarNode.from_file("data/grammar.xml")

for _ in range(20):
    leaf = root_node.random_walk()
    print(leaf)